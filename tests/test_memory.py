from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from original_llm.generate import (
    format_memory_block,
    parse_forget_command,
    parse_remember_command,
)
from original_llm.memory import (
    DEFAULT_IMPORTANCE,
    DEFAULT_MEMORY_DB_ENV,
    MAX_IMPORTANCE,
    MIN_IMPORTANCE,
    MemoryStore,
    default_memory_path,
)


class MemoryStoreCrudTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryStore(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def test_add_and_get_round_trip(self) -> None:
        entry = self.store.add("name", "夏樹")
        self.assertEqual(entry.key, "name")
        self.assertEqual(entry.value, "夏樹")
        self.assertEqual(entry.importance, DEFAULT_IMPORTANCE)
        self.assertTrue(entry.created_at)
        self.assertEqual(entry.created_at, entry.updated_at)

        fetched = self.store.get(entry.id)
        self.assertEqual(fetched, entry)

    def test_add_clamps_importance(self) -> None:
        self.assertEqual(self.store.add("k", "v", 0).importance, MIN_IMPORTANCE)
        self.assertEqual(self.store.add("k", "v", 9).importance, MAX_IMPORTANCE)

    def test_add_rejects_empty_value(self) -> None:
        with self.assertRaises(ValueError):
            self.store.add("k", "")
        with self.assertRaises(ValueError):
            self.store.add("", "v")

    def test_update_changes_value_and_timestamp(self) -> None:
        entry = self.store.add("hobby", "散歩", importance=4)
        updated = self.store.update(entry.id, value="読書")
        assert updated is not None
        self.assertEqual(updated.value, "読書")
        self.assertEqual(updated.importance, 4)
        self.assertGreaterEqual(updated.updated_at, entry.updated_at)

    def test_delete_returns_true_only_when_present(self) -> None:
        entry = self.store.add("k", "v")
        self.assertTrue(self.store.delete(entry.id))
        self.assertFalse(self.store.delete(entry.id))

    def test_delete_by_key_removes_all_matching(self) -> None:
        self.store.add("hobby", "散歩")
        self.store.add("hobby", "読書")
        self.store.add("name", "夏樹")
        deleted = self.store.delete_by_key("hobby")
        self.assertEqual(deleted, 2)
        remaining = self.store.list_all()
        self.assertEqual([e.key for e in remaining], ["name"])

    def test_clear_removes_all(self) -> None:
        self.store.add("a", "1")
        self.store.add("b", "2")
        deleted = self.store.clear()
        self.assertEqual(deleted, 2)
        self.assertEqual(self.store.list_all(), [])

    def test_list_all_orders_by_importance_then_recency(self) -> None:
        a = self.store.add("k1", "v1", importance=2)
        b = self.store.add("k2", "v2", importance=5)
        c = self.store.add("k3", "v3", importance=5)
        ordered = self.store.list_all()
        self.assertEqual(ordered[0].id, c.id)  # newest of importance 5
        self.assertEqual(ordered[1].id, b.id)
        self.assertEqual(ordered[2].id, a.id)


class MemoryStoreBumpOrAddTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryStore(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def test_bump_or_add_creates_first_entry_at_min_importance(self) -> None:
        entry = self.store.bump_or_add("word:ぴえん", "1")
        self.assertEqual(entry.key, "word:ぴえん")
        self.assertEqual(entry.value, "1")
        self.assertEqual(entry.importance, MIN_IMPORTANCE)

    def test_bump_or_add_raises_importance_and_updates_value(self) -> None:
        first = self.store.bump_or_add("word:ぴえん", "1")
        second = self.store.bump_or_add("word:ぴえん", "悲しい時の感情")
        self.assertEqual(second.id, first.id)
        self.assertEqual(second.value, "悲しい時の感情")
        self.assertEqual(second.importance, MIN_IMPORTANCE + 1)

    def test_bump_or_add_caps_importance(self) -> None:
        entry = self.store.bump_or_add("word:foo", "v")
        for _ in range(20):
            entry = self.store.bump_or_add("word:foo", "v")
        self.assertEqual(entry.importance, MAX_IMPORTANCE)


class MemoryStoreContainsWordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryStore(":memory:")
        self.store.add("name", "夏樹")
        self.store.add("word:ぴえん", "悲しい時の感情")

    def tearDown(self) -> None:
        self.store.close()

    def test_finds_value_match(self) -> None:
        self.assertTrue(self.store.contains_word("夏樹"))

    def test_finds_substring_in_value(self) -> None:
        self.assertTrue(self.store.contains_word("ぴえん"))

    def test_returns_false_when_unknown(self) -> None:
        self.assertFalse(self.store.contains_word("ピッツバーグ"))

    def test_empty_query_returns_false(self) -> None:
        self.assertFalse(self.store.contains_word(""))


class MemoryStoreRelevanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryStore(":memory:")
        self.name = self.store.add("name", "夏樹", importance=4)
        self.major = self.store.add("専攻", "計算機科学", importance=5)
        self.cat = self.store.add("ペット", "猫", importance=3)

    def tearDown(self) -> None:
        self.store.close()

    def test_no_query_returns_top_by_importance(self) -> None:
        result = self.store.select_relevant(query=None, limit=2)
        self.assertEqual([e.id for e in result], [self.major.id, self.name.id])

    def test_query_substring_in_key_boosts_score(self) -> None:
        # The user is asking about their major — the 専攻 entry should win.
        result = self.store.select_relevant("専攻なに？", limit=1)
        self.assertEqual(result[0].id, self.major.id)

    def test_query_substring_in_value_boosts_score(self) -> None:
        result = self.store.select_relevant("猫飼いたい", limit=1)
        self.assertEqual(result[0].id, self.cat.id)

    def test_limit_zero_returns_empty(self) -> None:
        self.assertEqual(self.store.select_relevant("anything", limit=0), [])


class MemoryPathResolutionTests(unittest.TestCase):
    def test_env_override_takes_precedence(self) -> None:
        with TemporaryDirectory() as tmp:
            override = os.path.join(tmp, "custom.db")
            old = os.environ.get(DEFAULT_MEMORY_DB_ENV)
            os.environ[DEFAULT_MEMORY_DB_ENV] = override
            try:
                self.assertEqual(default_memory_path(), Path(override))
            finally:
                if old is None:
                    os.environ.pop(DEFAULT_MEMORY_DB_ENV, None)
                else:
                    os.environ[DEFAULT_MEMORY_DB_ENV] = old

    def test_xdg_data_home_used_when_set(self) -> None:
        with TemporaryDirectory() as tmp:
            old_env = os.environ.get(DEFAULT_MEMORY_DB_ENV)
            old_xdg = os.environ.get("XDG_DATA_HOME")
            os.environ.pop(DEFAULT_MEMORY_DB_ENV, None)
            os.environ["XDG_DATA_HOME"] = tmp
            try:
                self.assertEqual(
                    default_memory_path(),
                    Path(tmp) / "original-llm" / "memory.db",
                )
            finally:
                if old_env is not None:
                    os.environ[DEFAULT_MEMORY_DB_ENV] = old_env
                if old_xdg is None:
                    os.environ.pop("XDG_DATA_HOME", None)
                else:
                    os.environ["XDG_DATA_HOME"] = old_xdg


class MemoryStorePersistenceTests(unittest.TestCase):
    def test_round_trip_through_disk(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "deep", "memory.db")
            with MemoryStore(db_path) as store:
                store.add("name", "夏樹", importance=4)
                store.add("専攻", "計算機科学", importance=5)
            self.assertTrue(os.path.exists(db_path))
            with MemoryStore(db_path) as reopened:
                entries = {e.key: e for e in reopened.list_all()}
                self.assertEqual(entries["name"].value, "夏樹")
                self.assertEqual(entries["専攻"].importance, 5)


class RememberCommandParserTests(unittest.TestCase):
    def test_basic_key_value(self) -> None:
        self.assertEqual(
            parse_remember_command(":remember name 夏樹"),
            ("name", "夏樹", DEFAULT_IMPORTANCE),
        )

    def test_equals_form(self) -> None:
        self.assertEqual(
            parse_remember_command(":remember 専攻=計算機科学"),
            ("専攻", "計算機科学", DEFAULT_IMPORTANCE),
        )

    def test_trailing_importance(self) -> None:
        self.assertEqual(
            parse_remember_command(":remember 専攻 計算機科学 5"),
            ("専攻", "計算機科学", 5),
        )

    def test_value_with_spaces(self) -> None:
        # Default behaviour: the first whitespace-separated token is the key,
        # the rest is the value.
        self.assertEqual(
            parse_remember_command(":remember note 今日は雨が降った 4"),
            ("note", "今日は雨が降った", 4),
        )

    def test_invalid_inputs_return_none(self) -> None:
        self.assertIsNone(parse_remember_command(":remember"))
        self.assertIsNone(parse_remember_command(":remember key"))
        self.assertIsNone(parse_remember_command(":notremember name 夏樹"))


class ForgetCommandParserTests(unittest.TestCase):
    def test_forget_id(self) -> None:
        self.assertEqual(parse_forget_command(":forget 12"), ("id", "12"))

    def test_forget_key(self) -> None:
        self.assertEqual(parse_forget_command(":forget-key hobby"), ("key", "hobby"))

    def test_forget_without_target(self) -> None:
        self.assertIsNone(parse_forget_command(":forget"))
        self.assertIsNone(parse_forget_command(":forget-key"))

    def test_unrelated_command(self) -> None:
        self.assertIsNone(parse_forget_command(":remember a b"))


class FormatMemoryBlockTests(unittest.TestCase):
    def test_empty_entries_returns_empty(self) -> None:
        self.assertEqual(format_memory_block([], "私", "相手"), "")

    def test_renders_chat_turn(self) -> None:
        store = MemoryStore(":memory:")
        try:
            store.add("name", "夏樹")
            store.add("専攻", "計算機科学")
            entries = store.list_all()
            block = format_memory_block(entries, "私", "相手")
            self.assertIn("私: 覚えておいてね:", block)
            self.assertIn("nameは夏樹", block)
            self.assertIn("専攻は計算機科学", block)
            self.assertIn("相手: うん、覚えた。", block)
            self.assertTrue(block.endswith("<eot>"))
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
