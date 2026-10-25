from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import respx

from deapi import AsyncDeapiClient, DeapiClient

BASE = "https://test.deapi.ai/api/v2"

KEY = {
    "id": 1234, "name": "ci", "preview": "sk-dp-1234-...abc", "role": "api", "status": "active",
    "created_at": "2026-01-01T00:00:00Z", "expires_at": None, "last_used_at": None, "ip_whitelist": [],
    "ip_whitelist_count": 0, "limit": None, "reset_interval": "none", "limit_used": None,
    "limit_remaining": None, "limit_resets_at": None,
}
USAGE = {"data": {"cost": 1.5, "requests": 10, "errors": 1, "groups": [{"key": "2026-01-01", "cost": 1.5}]}}


class TestAvailability:
    def test_v1_clients_have_no_account_or_keys(self, client: DeapiClient) -> None:
        assert not hasattr(client, "account")
        assert not hasattr(client, "keys")
        assert hasattr(client, "jobs")

    async def test_v1_async_clients_have_no_account_or_keys(self, async_client: AsyncDeapiClient) -> None:
        assert not hasattr(async_client, "account")
        assert not hasattr(async_client, "keys")

    def test_v2_has_them(self, client_v2: DeapiClient) -> None:
        assert client_v2.account is not None and client_v2.keys is not None

    @respx.mock
    def test_client_balance_still_works(self, client_v2: DeapiClient) -> None:
        respx.get(f"{BASE}/account/balance").mock(return_value=httpx.Response(200, json={"data": {"balance": 100.0}}))
        assert client_v2.balance().balance == 100.0


class TestAccountSync:
    @respx.mock
    def test_get(self, client_v2: DeapiClient) -> None:
        respx.get(f"{BASE}/account").mock(return_value=httpx.Response(200, json={"data": {
            "balance": 10.5, "currency": "USD", "auto_topup_enabled": True, "auto_topup_threshold": 5,
            "auto_topup_amount": 20, "future_field": 1}}))
        acc = client_v2.account.get()
        assert acc.balance == 10.5 and acc.currency == "USD" and acc.auto_topup_enabled is True
        assert acc.future_field == 1

    @respx.mock
    def test_balance(self, client_v2: DeapiClient) -> None:
        respx.get(f"{BASE}/account/balance").mock(return_value=httpx.Response(200, json={"data": {"balance": 3.0}}))
        assert client_v2.account.balance().balance == 3.0

    @respx.mock
    def test_usage_params(self, client_v2: DeapiClient) -> None:
        route = respx.get(f"{BASE}/account/usage").mock(return_value=httpx.Response(200, json=USAGE))
        rep = client_v2.account.usage(
            from_=datetime(2026, 1, 1, tzinfo=timezone.utc), to="2026-01-31", group_by="key", foo="bar",
        )
        p = route.calls.last.request.url.params
        assert p["from"] == "2026-01-01T00:00:00+00:00"
        assert p["to"] == "2026-01-31"
        assert p["group_by"] == "key" and p["foo"] == "bar"
        assert rep.cost == 1.5 and rep.requests == 10 and rep.errors == 1
        assert rep.groups[0]["key"] == "2026-01-01"

    @respx.mock
    def test_usage_list_payload(self, client_v2: DeapiClient) -> None:
        respx.get(f"{BASE}/account/usage").mock(return_value=httpx.Response(200, json={"data": [{"key": 1}]}))
        assert client_v2.account.usage().groups == [{"key": 1}]

    @respx.mock
    def test_usage_no_params(self, client_v2: DeapiClient) -> None:
        route = respx.get(f"{BASE}/account/usage").mock(return_value=httpx.Response(200, json=USAGE))
        client_v2.account.usage()
        assert dict(route.calls.last.request.url.params) == {}

    @respx.mock
    def test_payments(self, client_v2: DeapiClient) -> None:
        route = respx.get(f"{BASE}/account/payments").mock(return_value=httpx.Response(200, json={
            "data": [{"id": 1, "status": "paid", "amount": 20, "currency": "USD"}],
            "links": {"first": "x", "last": "y", "prev": None, "next": None},
            "meta": {"current_page": 1, "last_page": 1, "per_page": 15, "total": 1},
        }))
        page = client_v2.account.payments(page=2)
        assert route.calls.last.request.url.params["page"] == "2"
        assert page.data[0].status == "paid" and page.meta.total == 1


class TestKeysSync:
    @respx.mock
    def test_current(self, client_v2: DeapiClient) -> None:
        respx.get(f"{BASE}/key").mock(return_value=httpx.Response(200, json={"data": KEY}))
        k = client_v2.keys.current()
        assert k.id == 1234 and k.role == "api" and k.limit is None and k.ip_whitelist == []

    @respx.mock
    def test_current_usage(self, client_v2: DeapiClient) -> None:
        route = respx.get(f"{BASE}/key/usage").mock(return_value=httpx.Response(200, json=USAGE))
        rep = client_v2.keys.current_usage(group_by="model", from_="2026-01-01")
        assert route.calls.last.request.url.params["group_by"] == "model"
        assert route.calls.last.request.url.params["from"] == "2026-01-01"
        assert rep.cost == 1.5

    @respx.mock
    def test_list(self, client_v2: DeapiClient) -> None:
        route = respx.get(f"{BASE}/keys").mock(return_value=httpx.Response(200, json={
            "data": [KEY], "meta": {"current_page": 1, "last_page": 2, "per_page": 15, "total": 20}}))
        page = client_v2.keys.list(page=2)
        assert route.calls.last.request.url.params["page"] == "2"
        assert page.data[0].id == 1234 and page.meta.last_page == 2 and page.links is None

    @respx.mock
    def test_get_and_usage(self, client_v2: DeapiClient) -> None:
        respx.get(f"{BASE}/keys/5").mock(return_value=httpx.Response(200, json={"data": {**KEY, "id": 5}}))
        route = respx.get(f"{BASE}/keys/5/usage").mock(return_value=httpx.Response(200, json=USAGE))
        assert client_v2.keys.get(5).id == 5
        assert client_v2.keys.usage(5, to="2026-02-01").requests == 10
        assert route.calls.last.request.url.params["to"] == "2026-02-01"

    @respx.mock
    def test_create_exposes_secret(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{BASE}/keys").mock(return_value=httpx.Response(201, json={
            "data": {**KEY, "id": 9, "secret": "sk-dp-9-secret"}}))
        k = client_v2.keys.create(
            name="ci", expires_at=datetime(2027, 1, 1), limit=25.0, reset_interval="monthly",
            ip_whitelist=["1.2.3.4"],
        )
        assert json.loads(route.calls.last.request.content) == {
            "name": "ci", "expires_at": "2027-01-01T00:00:00", "limit": 25.0, "reset_interval": "monthly",
            "ip_whitelist": ["1.2.3.4"],
        }
        assert k.secret == "sk-dp-9-secret" and k.id == 9

    @respx.mock
    def test_create_minimal_body(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{BASE}/keys").mock(return_value=httpx.Response(201, json={"data": {**KEY, "secret": "s"}}))
        client_v2.keys.create(name="x")
        assert json.loads(route.calls.last.request.content) == {"name": "x"}

    @respx.mock
    def test_update_sends_only_given_fields(self, client_v2: DeapiClient) -> None:
        route = respx.patch(f"{BASE}/keys/5").mock(return_value=httpx.Response(200, json={
            "data": {**KEY, "id": 5, "status": "revoked"}}))
        k = client_v2.keys.update(5, status="revoked")
        assert json.loads(route.calls.last.request.content) == {"status": "revoked"}
        assert k.status == "revoked"

    @respx.mock
    def test_update_can_clear_fields(self, client_v2: DeapiClient) -> None:
        route = respx.patch(f"{BASE}/keys/5").mock(return_value=httpx.Response(200, json={"data": KEY}))
        client_v2.keys.update(5, limit=None, ip_whitelist=[], expires_at=None)
        assert json.loads(route.calls.last.request.content) == {
            "limit": None, "ip_whitelist": [], "expires_at": None}

    @respx.mock
    def test_delete(self, client_v2: DeapiClient) -> None:
        route = respx.delete(f"{BASE}/keys/5").mock(return_value=httpx.Response(200, json={"message": "deleted"}))
        assert client_v2.keys.delete(5) is None
        assert route.called


class TestAsync:
    @respx.mock
    async def test_account(self, async_client_v2: AsyncDeapiClient) -> None:
        respx.get(f"{BASE}/account").mock(return_value=httpx.Response(200, json={"data": {"balance": 1.0}}))
        respx.get(f"{BASE}/account/balance").mock(return_value=httpx.Response(200, json={"data": {"balance": 2.0}}))
        route = respx.get(f"{BASE}/account/usage").mock(return_value=httpx.Response(200, json=USAGE))
        respx.get(f"{BASE}/account/payments").mock(return_value=httpx.Response(200, json={"data": []}))
        assert (await async_client_v2.account.get()).balance == 1.0
        assert (await async_client_v2.account.balance()).balance == 2.0
        assert (await async_client_v2.account.usage(from_="2026-01-01", group_by="day")).cost == 1.5
        assert route.calls.last.request.url.params["group_by"] == "day"
        assert (await async_client_v2.account.payments()).data == []

    @respx.mock
    async def test_keys(self, async_client_v2: AsyncDeapiClient) -> None:
        respx.get(f"{BASE}/key").mock(return_value=httpx.Response(200, json={"data": KEY}))
        respx.get(f"{BASE}/key/usage").mock(return_value=httpx.Response(200, json=USAGE))
        respx.get(f"{BASE}/keys").mock(return_value=httpx.Response(200, json={"data": [KEY]}))
        respx.get(f"{BASE}/keys/5").mock(return_value=httpx.Response(200, json={"data": KEY}))
        respx.get(f"{BASE}/keys/5/usage").mock(return_value=httpx.Response(200, json=USAGE))
        create = respx.post(f"{BASE}/keys").mock(return_value=httpx.Response(201, json={
            "data": {**KEY, "secret": "sec"}}))
        patch = respx.patch(f"{BASE}/keys/5").mock(return_value=httpx.Response(200, json={"data": KEY}))
        delete = respx.delete(f"{BASE}/keys/5").mock(return_value=httpx.Response(200, json={"message": "ok"}))

        c = async_client_v2.keys
        assert (await c.current()).id == 1234
        assert (await c.current_usage()).requests == 10
        assert len((await c.list()).data) == 1
        assert (await c.get(5)).id == 1234
        assert (await c.usage(5)).errors == 1
        assert (await c.create(name="n", limit=5)).secret == "sec"
        assert json.loads(create.calls.last.request.content) == {"name": "n", "limit": 5}
        await c.update(5, status="active")
        assert json.loads(patch.calls.last.request.content) == {"status": "active"}
        assert await c.delete(5) is None
        assert delete.called
