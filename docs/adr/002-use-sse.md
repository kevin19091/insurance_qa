# ADR 002: Use SSE over WebSocket for streaming

**Status**: Accepted  
**Date**: 2026-06-01

## Context

The chat UI needs to display LLM tokens as they're generated (streaming). Two protocols: SSE (Server-Sent Events) and WebSocket.

## Decision

Use **SSE** over plain HTTP. Provide a separate `POST /api/chat/abort` endpoint for cancellation.

## Rationale

1. **Unidirectional by nature**: LLM response streaming is server → client. No client → server data during generation. Bidirectional WebSocket adds unnecessary complexity.
2. **Simpler connection lifecycle**: SSE is plain HTTP. No upgrade handshake, no heartbeats, no reconnection logic. Browser `EventSource` works natively.
3. **Abort is a separate concern**: User clicking "Stop" is a separate request (`POST /api/chat/abort`) that cancels the server-side task. This is cleaner than an in-band WebSocket frame.
4. **Single connection model**: With a single Railway VM, keeping things simple matters. SSE + HTTP is easier to debug and monitor.

## Consequences

- Abort requires two endpoints instead of one WebSocket
- Not suitable if future scope includes bidirectional real-time (collaborative chat, etc.) — reconsider then
- `EventSource` is available in all modern browsers
