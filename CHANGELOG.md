# Changelog

All notable changes to this project are documented in this file.
Versions follow [Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may contain breaking changes).

## [0.6.0] - 2026-11-19

### Added

- `client.websocket_config()` (sync and async) returns the account's realtime connection settings
  (`GET /websocket`) as the new `WebSocketConfig` / `WebSocketConnection` models. Works with any API key.

### Changed

- `client.realtime()` no longer needs any settings. `client_id`, `ws_url` and `app_key` are each taken
  from the argument, then `DEAPI_CLIENT_ID` / `DEAPI_WS_URL` / `DEAPI_WS_APP_KEY`, then `GET /websocket`.
  The discovery call is made once at connect time and only when something is still missing; with all
  three set, no extra request is made.
- Setting only `DEAPI_CLIENT_ID` now triggers one `GET /websocket` call at connect time instead of
  falling back to built-in WebSocket defaults. The hard-coded WebSocket URL and app key were removed.
- `Realtime.client_id` / `AsyncRealtime.client_id` is now `str | None`: `None` until connect when it was
  neither passed nor set in the environment.
- An unusable discovery response raises `RealtimeError`; HTTP errors from it raise the usual API exceptions.

## Earlier releases

See the [tags](https://github.com/deapi-ai/deapi-python-sdk/tags) for 0.5.2 and older.
