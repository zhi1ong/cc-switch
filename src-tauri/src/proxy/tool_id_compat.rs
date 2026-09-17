//! Anthropic tool_use ID 冲突兼容（响应侧改写）
//!
//! 背景：部分 Anthropic 协议兼容上游（实测：百炼/DashScope 托管的 kimi-k3）
//! 每轮重置 tool_use ID，如 `toolu_Bash_0` 每轮从头编号。当响应里的
//! tool_use ID 与会话历史中的 ID 重复时，Claude Code / Claude Desktop 会
//! 执行工具但在构造下一轮请求时丢弃重复 ID 对应的 tool_use / tool_result，
//! 模型看不到结果，只能反复请求同一调用，形成工具调用循环。
//!
//! 兼容策略：从客户端请求的消息历史中收集已出现的 tool ID
//! （`tool_use.id` 与 `tool_result.tool_use_id`）；响应侧发现 tool_use
//! ID 与历史冲突或在同一响应内重复时，替换为新的唯一 ID。客户端下一轮
//! 用新 ID 回写 assistant `tool_use` 与 `tool_result`，配对一致，循环
//! 消失。这类上游本就每轮重置 ID，历史里的新 ID 直接透传即可，无需还原。
//!
//! 成本控制：检测按需触发——非流式响应先用 `"tool_use"` 子串门禁，SSE
//! 逐事件门禁，只有真正出现 tool_use 块才解析 JSON，只有真正冲突才改写。
//! 正常上游零改写，字节级原样透传。流式路径以独立的流包装器实现
//! （复用 `sse::SseDecoder` 增量分帧，与 `response_model_rewriter` 同构），
//! 与 usage 收集、模型回写互不影响。

use super::log_codes;
use super::sse::{sse_data, sse_lines, strip_sse_field, SseDecoder, SseFrame};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// tool ID 冲突改写器
///
/// 一次请求对应一个实例：`from_request` 预留历史 ID，响应处理阶段
/// （流式逐 SSE 事件 / 非流式整包）调用改写方法。
pub(crate) struct ToolIdRewriter {
    /// 已见过的 tool ID：请求历史 + 本响应已分配的 ID。
    seen: HashSet<String>,
    /// 流式：content block index → (上游原始 ID, 分配给客户端的 ID)。
    /// 上游重发同一 index 的 block start 时保持已分配 ID 稳定。
    blocks: HashMap<u64, ToolBlockAssignment>,
    /// 实际发生的改写次数（日志用）。
    rewrites: usize,
}

struct ToolBlockAssignment {
    original: String,
    client: String,
}

impl ToolIdRewriter {
    /// 从客户端请求体收集会话历史中已出现的 tool ID。
    ///
    /// 只做一次只读遍历，不修改请求体。
    pub(crate) fn from_request(body: &Value) -> Self {
        let mut seen = HashSet::new();
        if let Some(messages) = body.get("messages").and_then(Value::as_array) {
            for message in messages {
                let Some(content) = message.get("content").and_then(Value::as_array) else {
                    // content 为字符串（纯文本消息）或缺失时无需处理
                    continue;
                };
                for block in content {
                    match block.get("type").and_then(Value::as_str) {
                        Some("tool_use") => {
                            if let Some(id) = block.get("id").and_then(Value::as_str) {
                                reserve_id(&mut seen, id);
                            }
                        }
                        Some("tool_result") => {
                            if let Some(id) = block.get("tool_use_id").and_then(Value::as_str) {
                                reserve_id(&mut seen, id);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        Self {
            seen,
            blocks: HashMap::new(),
            rewrites: 0,
        }
    }

    /// 实际发生的改写次数。
    pub(crate) fn rewrites(&self) -> usize {
        self.rewrites
    }

    /// 非流式响应改写：`content` 数组中与历史冲突或在包内重复的
    /// `tool_use.id` 替换为唯一 ID。
    ///
    /// 未发生冲突时返回 `None`（调用方保持字节级原样透传）。
    pub(crate) fn rewrite_buffered(&mut self, body: &[u8]) -> Option<Vec<u8>> {
        // 子串门禁：无 tool_use 块的响应（纯文本/错误体）直接跳过 JSON 解析
        let text = std::str::from_utf8(body).ok()?;
        if !text.contains("\"tool_use\"") {
            return None;
        }
        let mut value = serde_json::from_str::<Value>(text).ok()?;

        let mut changed = false;
        if let Some(content) = value.get_mut("content").and_then(Value::as_array_mut) {
            for block in content {
                // 只处理客户端工具块：server_tool_use 的 ID 由上游自行匹配，
                // 改写会破坏配对
                if block.get("type").and_then(Value::as_str) != Some("tool_use") {
                    continue;
                }
                let Some(id) = block
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty())
                    .map(str::to_string)
                else {
                    continue;
                };
                let client_id = self.unique_id(&id);
                if client_id != id {
                    if let Some(block) = block.as_object_mut() {
                        block.insert("id".to_string(), Value::String(client_id));
                        changed = true;
                    }
                }
            }
        }

        if !changed {
            return None;
        }
        serde_json::to_vec(&value).ok()
    }

    /// 流式响应改写：单个完整 SSE 事件块中的 `content_block_start`
    /// (tool_use) 冲突 ID 替换为唯一 ID。
    ///
    /// `block` 是完整事件文本，允许包含 LF / CRLF / CR 结尾空行。
    /// 未发生冲突时返回 `None`。
    pub(crate) fn rewrite_sse_block(&mut self, block: &str) -> Option<String> {
        if !block.contains("\"tool_use\"") {
            return None;
        }

        // SSE 的多行 data 共同组成一个 JSON 载荷，必须合并后再解析。
        let data = sse_data(block)?;
        let mut event = serde_json::from_str::<Value>(&data).ok()?;
        if event.get("type").and_then(Value::as_str) != Some("content_block_start") {
            return None;
        }
        let content_block = event.get("content_block")?;
        if content_block.get("type").and_then(Value::as_str) != Some("tool_use") {
            return None;
        }
        let id = content_block
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())?
            .to_string();
        // 缺失 index 时无法为重发的 block start 保持身份稳定，跳过
        let index = event.get("index").and_then(Value::as_u64)?;

        let client_id = match self.blocks.get(&index) {
            Some(assignment) if assignment.original == id => assignment.client.clone(),
            _ => {
                let client = self.unique_id(&id);
                self.blocks.insert(
                    index,
                    ToolBlockAssignment {
                        original: id.clone(),
                        client: client.clone(),
                    },
                );
                client
            }
        };
        if client_id == id {
            return None;
        }

        event
            .get_mut("content_block")?
            .as_object_mut()?
            .insert("id".to_string(), Value::String(client_id));
        let serialized = serde_json::to_string(&event).ok()?;

        // 只替换 data 字段，保留事件名、ID、注释和 BOM。把合并后的载荷放在
        // 最后一个 data 行，保留其结尾，避免混合 CR/LF 在删行后合成 CRLF。
        let mut remaining_data = sse_lines(block)
            .enumerate()
            .filter(|(index, (line, _))| {
                let field = if *index == 0 {
                    line.strip_prefix('\u{feff}').unwrap_or(line)
                } else {
                    line
                };
                strip_sse_field(field, "data").is_some()
            })
            .count();
        let mut rewritten = String::with_capacity(block.len());
        for (index, (line, ending)) in sse_lines(block).enumerate() {
            let field = if index == 0 {
                line.strip_prefix('\u{feff}').unwrap_or(line)
            } else {
                line
            };
            if strip_sse_field(field, "data").is_some() {
                if field.len() != line.len() {
                    rewritten.push('\u{feff}');
                }
                remaining_data -= 1;
                if remaining_data == 0 {
                    rewritten.push_str("data: ");
                    rewritten.push_str(&serialized);
                    rewritten.push_str(ending);
                }
            } else {
                rewritten.push_str(line);
                rewritten.push_str(ending);
            }
        }
        Some(rewritten)
    }

    /// 返回可用的 ID：未冲突时原样返回并预留；冲突时生成新的
    /// `toolu_<uuid>`（无连字符，贴近 Anthropic 真实 ID 形态）。
    fn unique_id(&mut self, id: &str) -> String {
        if !self.seen.contains(id) {
            self.seen.insert(id.to_string());
            return id.to_string();
        }
        loop {
            let candidate = format!("toolu_{}", Uuid::new_v4().simple());
            if !self.seen.contains(&candidate) {
                self.seen.insert(candidate.clone());
                self.rewrites += 1;
                return candidate;
            }
        }
    }
}

fn reserve_id(seen: &mut HashSet<String>, id: &str) {
    if !id.is_empty() {
        seen.insert(id.to_string());
    }
}

/// 包装一个字节流，逐个 SSE 事件改写冲突的 tool_use ID。
///
/// 与 `response_model_rewriter::create_model_rewriting_stream` 同构：按 SSE
/// 的 LF / CRLF / CR 行结束规则增量分帧（复用 `SseDecoder`）。事件前的注释
/// 心跳即时透传；data 事件等待结尾空行后按需改写。未修改的事件复用原始
/// `Bytes`（零拷贝），跨 chunk 事件只缓冲一次。正常 EOF 的尾部原样透传。
fn create_tool_id_rewriting_stream(
    stream: impl Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static,
    mut rewriter: ToolIdRewriter,
    tag: &'static str,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    async_stream::stream! {
        let mut decoder = SseDecoder::default();

        tokio::pin!(stream);

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    decoder.push(bytes);
                    while let Some(frame) = decoder.next_frame() {
                        yield Ok(match frame {
                            SseFrame::Passthrough(bytes) => bytes,
                            SseFrame::Event(bytes) => match std::str::from_utf8(&bytes) {
                                Ok(block) => match rewriter.rewrite_sse_block(block) {
                                    Some(rewritten) => Bytes::from(rewritten),
                                    None => bytes,
                                },
                                // 无效 UTF-8 不做有损改写，保留上游原始字节
                                Err(_) => bytes,
                            },
                        });
                    }
                }
                Err(e) => {
                    yield Err(e);
                    return;
                }
            }
        }

        // 流正常结束：冲刷未终止的尾部残余，原样透传
        let tail = decoder.finish();
        if !tail.is_empty() {
            yield Ok(tail);
        }
        if rewriter.rewrites() > 0 {
            log::info!(
                "[{tag}] [{}] 检测到 tool_use ID 与历史冲突，已替换 {} 处（上游每轮重置 tool ID 的兼容处理）",
                log_codes::rsp::TOOL_ID_REWRITTEN,
                rewriter.rewrites()
            );
        }
    }
}

/// 便捷包装：有改写器时用 [`create_tool_id_rewriting_stream`] 包装，否则
/// 原样返回。统一装箱，让调用点的两种分支类型一致。
pub(crate) fn wrap_stream_for_tool_id_rewrite(
    stream: impl Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static,
    rewriter: Option<ToolIdRewriter>,
    tag: &'static str,
) -> std::pin::Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>> {
    match rewriter {
        Some(rewriter) => Box::pin(create_tool_id_rewriting_stream(stream, rewriter, tag)),
        None => Box::pin(stream),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const HISTORY: &str = r#"{"messages":[
        {"role":"assistant","content":[{"type":"tool_use","id":"toolu_Bash_0","name":"Bash","input":{"command":"first"}}]},
        {"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_Bash_0","content":"first-result"}]}
    ]}"#;

    fn rewriter_from_history() -> ToolIdRewriter {
        ToolIdRewriter::from_request(&serde_json::from_str(HISTORY).unwrap())
    }

    #[test]
    fn from_request_collects_tool_use_and_tool_result_ids() {
        let body: Value = serde_json::from_str(HISTORY).unwrap();
        let rewriter = ToolIdRewriter::from_request(&body);
        assert!(rewriter.seen.contains("toolu_Bash_0"));
        assert_eq!(rewriter.seen.len(), 1);
    }

    #[test]
    fn from_request_ignores_text_content_and_empty_ids() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "plain string content"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "toolu_Bash_0 is literal text"},
                    {"type": "tool_use", "id": "", "name": "Bash", "input": {}},
                    {"type": "tool_use", "name": "NoId", "input": {}},
                    {"type": "tool_result", "tool_use_id": "result_id"}
                ]}
            ]
        });
        let rewriter = ToolIdRewriter::from_request(&body);
        assert!(rewriter.seen.contains("result_id"));
        assert_eq!(rewriter.seen.len(), 1);
    }

    #[test]
    fn from_request_handles_missing_messages() {
        let rewriter = ToolIdRewriter::from_request(&json!({"model": "claude"}));
        assert!(rewriter.seen.is_empty());
    }

    #[test]
    fn rewrite_buffered_only_touches_conflicting_client_tool_id() {
        let body = r#"{"id":"msg_2","model":"kimi-k3","content":[
            {"type":"thinking","thinking":"toolu_Bash_0 is literal text","signature":"opaque"},
            {"type":"tool_use","id":"toolu_Bash_0","name":"Bash","input":{"command":"echo toolu_Bash_0"}},
            {"type":"tool_use","id":"toolu_Write_1","name":"Write","input":{"content":"toolu_Bash_0"}},
            {"type":"server_tool_use","id":"toolu_Bash_0","name":"web_search","input":{}}
        ],"stop_reason":"tool_use","usage":{"input_tokens":12,"output_tokens":34}}"#;
        let mut rewriter = rewriter_from_history();
        let got = rewriter
            .rewrite_buffered(body.as_bytes())
            .expect("应发生改写");

        let parsed: Value = serde_json::from_slice(&got).unwrap();
        let new_id = parsed["content"][1]["id"].as_str().unwrap().to_string();
        assert!(new_id.starts_with("toolu_"));
        assert_ne!(new_id, "toolu_Bash_0");
        assert_eq!(parsed["content"][1]["name"], "Bash");
        // 其余字段一律不动
        assert_eq!(
            parsed["content"][0]["thinking"],
            "toolu_Bash_0 is literal text"
        );
        assert_eq!(parsed["content"][2]["id"], "toolu_Write_1");
        assert_eq!(parsed["content"][2]["input"]["content"], "toolu_Bash_0");
        assert_eq!(parsed["content"][3]["id"], "toolu_Bash_0");
        assert_eq!(parsed["id"], "msg_2");
        assert_eq!(rewriter.rewrites(), 1);
    }

    #[test]
    fn rewrite_buffered_reserves_ids_within_response() {
        let body = r#"{"content":[
            {"type":"tool_use","id":"same","name":"Read","input":{"file_path":"first"}},
            {"type":"tool_use","id":"same","name":"Read","input":{"file_path":"second"}},
            {"type":"tool_use","id":"same","name":"Read","input":{"file_path":"third"}}
        ]}"#;
        let mut rewriter = ToolIdRewriter::from_request(&json!({}));
        let got = rewriter
            .rewrite_buffered(body.as_bytes())
            .expect("应发生改写");
        let parsed: Value = serde_json::from_slice(&got).unwrap();
        let ids: Vec<&str> = parsed["content"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids[0], "same");
        assert_ne!(ids[0], ids[1]);
        assert_ne!(ids[0], ids[2]);
        assert_ne!(ids[1], ids[2]);
    }

    #[test]
    fn rewrite_buffered_reserves_tool_result_history() {
        let body = json!({"content": [
            {"type": "tool_use", "id": "toolu_Bash_0", "name": "Bash", "input": {}}
        ]});
        let history = json!({"messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_Bash_0", "content": "result"}
            ]}
        ]});
        let mut rewriter = ToolIdRewriter::from_request(&history);
        let got = rewriter.rewrite_buffered(serde_json::to_string(&body).unwrap().as_bytes());
        assert!(got.is_some());
    }

    #[test]
    fn rewrite_buffered_leaves_unrelated_bodies_untouched() {
        for body in [
            r#" { "content": [ { "type": "tool_use", "id": "unique", "name": "Read", "input": {} } ] } "#,
            r#"{"content":[{"type":"text","text":"toolu_Bash_0"}]}"#,
            r#"{"content":[{"type":"tool_use","id":"","name":"Read","input":{}}]}"#,
            r#"{"content":[{"type":"tool_use","name":"Read","input":{}}]}"#,
            r#"{"type":"error","error":{"type":"api_error","message":"tool_use failed"}}"#,
            "not json",
        ] {
            let mut rewriter = rewriter_from_history();
            assert_eq!(
                rewriter.rewrite_buffered(body.as_bytes()),
                None,
                "body: {body}"
            );
        }
    }

    #[test]
    fn rewrite_sse_block_rewrites_conflicting_tool_use_start() {
        let mut rewriter = rewriter_from_history();
        let block = "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_Bash_0\",\"name\":\"Bash\",\"input\":{}}}";
        let got = rewriter.rewrite_sse_block(block).expect("应发生改写");
        assert!(got.starts_with("event: content_block_start\ndata: "));
        let parsed: Value = serde_json::from_str(got.split_once("data: ").unwrap().1).unwrap();
        let new_id = parsed["content_block"]["id"].as_str().unwrap();
        assert!(new_id.starts_with("toolu_"));
        assert_ne!(new_id, "toolu_Bash_0");
        assert_eq!(parsed["content_block"]["name"], "Bash");
        assert_eq!(rewriter.rewrites(), 1);
    }

    #[test]
    fn rewrite_sse_block_keeps_identity_for_repeated_start() {
        let mut rewriter = rewriter_from_history();
        let block = "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_Bash_0\",\"name\":\"Bash\",\"input\":{}}}";
        let first = rewriter.rewrite_sse_block(block).expect("应发生改写");
        // 上游重发同一 index 的 block start：保持已分配的客户端 ID
        let second = rewriter.rewrite_sse_block(block).expect("应再次改写");
        assert_eq!(first, second);
        assert_eq!(rewriter.rewrites(), 1);
    }

    #[test]
    fn rewrite_sse_block_distinguishes_parallel_blocks_with_same_id() {
        let mut rewriter = rewriter_from_history();
        let start = "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_Bash_0\",\"name\":\"Bash\",\"input\":{}}}";
        let first = rewriter.rewrite_sse_block(start).unwrap();
        let parallel = start.replace("\"index\":1", "\"index\":2");
        let second = rewriter.rewrite_sse_block(&parallel).unwrap();

        let id_of = |text: &str| {
            serde_json::from_str::<Value>(text.split_once("data: ").unwrap().1).unwrap()
                ["content_block"]["id"]
                .as_str()
                .unwrap()
                .to_string()
        };
        assert_ne!(id_of(&first), "toolu_Bash_0");
        assert_ne!(id_of(&second), "toolu_Bash_0");
        assert_ne!(id_of(&first), id_of(&second));
    }

    #[test]
    fn rewrite_sse_block_handles_spaced_json() {
        let mut rewriter = rewriter_from_history();
        let block = "data: {\"type\": \"content_block_start\", \"index\": 1, \"content_block\": {\"type\": \"tool_use\", \"id\": \"toolu_Bash_0\", \"name\": \"Bash\", \"input\": {}}}";
        let got = rewriter
            .rewrite_sse_block(block)
            .expect("宽松 JSON 也应改写");
        let parsed: Value =
            serde_json::from_str(got.split_once("data: ").unwrap().1).expect("改写行仍是合法 JSON");
        assert_ne!(parsed["content_block"]["id"], "toolu_Bash_0");
        assert!(parsed["content_block"]["id"]
            .as_str()
            .unwrap()
            .starts_with("toolu_"));
    }

    #[test]
    fn rewrite_sse_block_leaves_unrelated_events_untouched() {
        let mut rewriter = rewriter_from_history();
        for block in [
            "event: content_block_start",
            "data: {\"type\":\"content_block_start\",\"index\":3,\"content_block\":{\"type\":\"tool_use\",\"id\":\"unique\",\"name\":\"Read\",\"input\":{}}}",
            "data: {\"type\":\"content_block_start\",\"index\":4,\"content_block\":{\"type\":\"server_tool_use\",\"id\":\"toolu_Bash_0\",\"name\":\"web_search\",\"input\":{}}}",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"command\\\":\\\"echo toolu_Bash_0\\\"}\"}}",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"opaque\"}}",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":42}}",
            "data: {\"type\":\"error\",\"error\":{\"message\":\"toolu_Bash_0\"}}",
            "data: [DONE]",
            "data: broken",
            "",
        ] {
            assert_eq!(rewriter.rewrite_sse_block(block), None, "block: {block}");
        }
    }

    // ------------------------------------------------------------------
    // 流包装器测试（SseDecoder 分帧 + 改写）
    // ------------------------------------------------------------------

    const TOOL_CONFLICT_SSE: &str = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_2\",\"model\":\"kimi-k3\"}}\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_Bash_0\",\"name\":\"Bash\",\"input\":{}}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"command\\\":\\\"second\\\"}\"}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";

    async fn wrapped_output(chunks: Vec<&str>, history: &Value) -> String {
        let input: Vec<Result<Bytes, std::io::Error>> = chunks
            .into_iter()
            .map(|c| Ok(Bytes::from(c.to_string())))
            .collect();
        let stream = wrap_stream_for_tool_id_rewrite(
            futures::stream::iter(input),
            Some(ToolIdRewriter::from_request(history)),
            "test",
        );
        let mut out = String::new();
        let mut items = std::pin::pin!(stream);
        while let Some(item) = items.next().await {
            out.push_str(&String::from_utf8_lossy(&item.unwrap()));
        }
        out
    }

    fn parse_sse_events(output: &str) -> Vec<Value> {
        let mut decoder = SseDecoder::default();
        decoder.push(Bytes::copy_from_slice(output.as_bytes()));
        let mut events = Vec::new();
        while let Some(frame) = decoder.next_frame() {
            if let SseFrame::Event(bytes) = frame {
                let data = sse_data(std::str::from_utf8(&bytes).unwrap()).unwrap();
                events.push(serde_json::from_str(&data).unwrap());
            }
        }
        assert!(
            decoder.finish().is_empty(),
            "SSE must end with a complete event"
        );
        events
    }

    #[tokio::test]
    async fn stream_rewrites_conflicts_with_all_line_endings_and_chunk_boundaries() {
        let history: Value = serde_json::from_str(HISTORY).unwrap();
        for ending in ["\n", "\r\n", "\r"] {
            let input = TOOL_CONFLICT_SSE.replace('\n', ending);
            for size in [1, 2, 7, 4096] {
                let chunks = (0..input.len())
                    .step_by(size)
                    .map(|start| &input[start..(start + size).min(input.len())])
                    .collect();
                let out = wrapped_output(chunks, &history).await;
                let events = parse_sse_events(&out);
                let mut expected = parse_sse_events(&input);
                let id = events[1]["content_block"]["id"].as_str().unwrap();
                assert_ne!(id, "toolu_Bash_0");
                assert!(id.starts_with("toolu_"));
                expected[1]["content_block"]["id"] = json!(id);
                assert_eq!(events, expected);
            }
        }
    }

    #[tokio::test]
    async fn stream_rewrites_multiline_data_with_mixed_line_endings() {
        let history: Value = serde_json::from_str(HISTORY).unwrap();
        for ending in ["\n", "\r\n", "\r"] {
            for delimiter in ["\n\n", "\r\n\r\n", "\r\r", "\r\n\n", "\n\r"] {
                let input = format!(
                    "event: content_block_start{ending}data: {{\"type\":\"content_block_start\",\"index\":0,{ending}id: event-1\r: keep metadata\rdata: \"content_block\":{{\"type\":\"tool_use\",\"id\":\"toolu_Bash_0\",\"name\":\"Bash\",\"input\":{{\"command\":\"echo toolu_Bash_0\"}}}}}}{delimiter}event: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n"
                );
                for size in [1, 7, 4096] {
                    let chunks: Vec<_> = (0..input.len())
                        .step_by(size)
                        .map(|start| &input[start..(start + size).min(input.len())])
                        .collect();
                    // 无冲突时，多行载荷和所有元数据必须保持原始字节。
                    assert_eq!(wrapped_output(chunks.clone(), &json!({})).await, input);
                    let out = wrapped_output(chunks, &history).await;
                    assert!(out.contains("id: event-1\r: keep metadata\r"));
                    assert!(out.contains(&format!("{delimiter}event: message_stop\n")));
                    let events = parse_sse_events(&out);
                    let mut expected = parse_sse_events(&input);
                    let id = events[0]["content_block"]["id"].as_str().unwrap();
                    assert_ne!(id, "toolu_Bash_0");
                    expected[0]["content_block"]["id"] = json!(id);
                    assert_eq!(events, expected);
                }
            }
        }
    }

    #[test]
    fn rewrite_sse_block_preserves_bom_and_metadata_matching_payload() {
        let payload = r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_Bash_0","name":"Bash","input":{}}}"#;
        let block = format!("\u{feff}data: {payload}\r: {payload}\rid: {payload}\r\r");
        let mut rewriter = rewriter_from_history();
        let out = rewriter.rewrite_sse_block(&block).unwrap();
        assert!(out.starts_with("\u{feff}data: "));
        assert!(out.ends_with(&format!("\r: {payload}\rid: {payload}\r\r")));
        let event: Value = serde_json::from_str(&sse_data(&out).unwrap()).unwrap();
        assert_ne!(event["content_block"]["id"], "toolu_Bash_0");
    }

    #[tokio::test]
    async fn stream_without_conflict_is_byte_identical() {
        // 无冲突：输出必须与上游字节流完全一致（含定界符与尾部）
        let out = wrapped_output(vec!["data: 1", "2\n\ndata: 3\n\n"], &json!({})).await;
        assert_eq!(out, "data: 12\n\ndata: 3\n\n");
    }

    #[tokio::test]
    async fn stream_without_rewriter_returns_input_unchanged() {
        // 无改写器（开关关闭）：包装器直接原样返回
        let input: Vec<Result<Bytes, std::io::Error>> = vec![
            Ok(Bytes::from("data: partial")),
            Ok(Bytes::from(" event\n\n")),
        ];
        let stream = wrap_stream_for_tool_id_rewrite(futures::stream::iter(input), None, "test");
        let out: Vec<_> = stream.collect().await;
        let joined = out
            .into_iter()
            .map(|b| String::from_utf8_lossy(&b.unwrap()).to_string())
            .collect::<String>();
        assert_eq!(joined, "data: partial event\n\n");
    }

    #[tokio::test]
    async fn stream_preserves_crlf_and_cr_line_endings() {
        for ending in ["\r\n", "\r"] {
            let input = format!(
                "event: message_start{ending}data: {{\"type\":\"message_start\"}}{ending}{ending}data: {{\"type\":\"message_stop\"}}{ending}{ending}"
            );
            let out = wrapped_output(vec![&input], &json!({})).await;
            assert_eq!(out, input);
        }
    }

    #[tokio::test]
    async fn stream_rewrites_only_conflicting_tool_event() {
        let history: Value = serde_json::from_str(HISTORY).unwrap();
        let out = wrapped_output(vec![TOOL_CONFLICT_SSE], &history).await;

        // 冲突事件被改写
        assert!(!out.contains("\"id\":\"toolu_Bash_0\""));
        // 其余事件字节级不变
        assert!(out.contains("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_2\",\"model\":\"kimi-k3\"}}\n\n"));
        assert!(out.contains("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"command\\\":\\\"second\\\"}\"}}\n\n"));
        assert!(out.contains("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));
        // 结构保持：四类事件齐全
        assert_eq!(out.matches("\n\n").count(), 4);
    }

    #[tokio::test]
    async fn stream_handles_event_split_across_chunks() {
        // 事件跨 chunk 拆分：块仍完整改写，整体输出与单 chunk 模式一致
        let history: Value = serde_json::from_str(HISTORY).unwrap();
        let sse = TOOL_CONFLICT_SSE;
        let split = sse.find("\"index\":0").unwrap();
        let (head, tail) = sse.split_at(split);
        let out = wrapped_output(vec![head, tail], &history).await;
        let single = wrapped_output(vec![sse], &history).await;
        // 新生成的 tool ID 为随机值，归一化后比较
        let normalize = |text: &str| {
            let (prefix, rest) = text
                .split_once("\"id\":\"toolu_")
                .expect("应包含改写后的 tool ID");
            let suffix = rest.split_once('"').map(|(_, s)| s).unwrap_or("");
            format!("{prefix}<TOOLU>{suffix}")
        };
        assert_eq!(normalize(&out), normalize(&single));
    }

    #[tokio::test]
    async fn stream_flushes_trailing_partial_bytes() {
        // 上游末尾未补定界符：尾部字节原样冲刷，不丢数据
        let out = wrapped_output(
            vec!["data: {\"type\":\"message_stop\"}\n\ndata: partial-tail"],
            &json!({}),
        )
        .await;
        assert_eq!(
            out,
            "data: {\"type\":\"message_stop\"}\n\ndata: partial-tail"
        );
    }

    #[tokio::test]
    async fn stream_passes_non_sse_body_unchanged() {
        // 响应头是 SSE 但 body 实际不是事件流（异常上游）：原样转发
        let out = wrapped_output(vec!["not an event stream at all"], &json!({})).await;
        assert_eq!(out, "not an event stream at all");
    }

    #[tokio::test]
    async fn stream_passes_heartbeat_comments_through() {
        // SSE 注释心跳与事件间杂：全部字节级透传（无冲突时）
        let input = ": keep-alive\n\ndata: {\"type\":\"ping\"}\n\n: keep-alive\n\n";
        let out = wrapped_output(vec![input], &json!({})).await;
        assert_eq!(out, input);
    }
}
