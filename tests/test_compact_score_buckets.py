"""Group by score in compact mode (`compact_score_buckets`, off by default).

The band is cut into blocks of `compact_bucket_size` numbers, best whole-star
score first (★9-10, ★8 ... ★1, ★0 open-ended to the end of the band). What it
must NOT do is what #117 / #121 / #310 declined "order by rating" for: move a
published game. The guide binds by the bare channel number, so a game is
placed by its score ONCE, when it is first numbered, and holds that number
however its score changes afterwards.

Assertions use LITERAL numbers (400, 410, 490) for the reason
test_compact_numbering.py gives: re-deriving `base + i * size` in the test
stays green when the production arithmetic is wrong.

The apply scenarios drive the real `_action_apply` against a real Django ORM
(tests/orm_harness/run_scenarios.py), as test_favorites_first.py does.
"""

import importlib.util
import json
import os
import random
from datetime import datetime, timedelta, timezone

import pytest

from dispatcharr_ranked_matchups._util import (
    allocate_compact_blocks,
    allocate_compact_numbers,
)

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------- allocate_compact_blocks ---

class TestBlocks:
    def test_each_game_goes_to_its_block_in_the_given_order(self):
        got = allocate_compact_blocks(
            ["a", "b", "c", "d"], {"a": 3, "b": 0, "c": 3, "d": 9}, 400, 100, 10,
        )
        assert got == {"b": 400, "a": 430, "c": 431, "d": 490}

    def test_the_last_block_is_open_ended_to_the_end_of_the_band(self):
        markers = [f"z{i:02d}" for i in range(15)]
        got = allocate_compact_blocks(markers, {m: 9 for m in markers}, 400, 120, 10)
        assert got["z00"] == 490
        assert got["z14"] == 504
        assert max(got.values()) == 504

    def test_a_full_block_spills_down_never_up(self):
        # Block 1 (410-419) is full of published games; the new ★8 game must
        # not take a free number in block 0 (400-409), it goes to block 2.
        existing = {f"p{i}": 410 + i for i in range(10)}
        markers = list(existing) + ["new"]
        blocks = {m: 1 for m in markers}
        got = allocate_compact_blocks(markers, blocks, 400, 100, 10, existing)
        assert got["new"] == 420
        assert all(n >= 410 for n in got.values())

    def test_spilled_games_go_ahead_of_the_lower_blocks_own_games(self):
        # 3-number blocks: four ★9-10 games, two ★8 games. The fourth top game
        # spills into block 1 ahead of block 1's own games, which then spill on.
        got = allocate_compact_blocks(
            ["t1", "t2", "t3", "t4", "e1", "e2"],
            {"t1": 0, "t2": 0, "t3": 0, "t4": 0, "e1": 1, "e2": 1},
            400, 30, 3,
        )
        assert got == {"t1": 400, "t2": 401, "t3": 402, "t4": 403, "e1": 404, "e2": 405}

    def test_spill_continues_anywhere_below(self):
        # Blocks 1 and 2 are full; a block-1 game lands in block 3.
        existing = {f"p{i}": 403 + i for i in range(6)}
        markers = list(existing) + ["new"]
        got = allocate_compact_blocks(markers, {m: 1 for m in markers}, 400, 30, 3, existing)
        assert got["new"] == 409

    def test_nothing_fits_below_means_omitted_not_moved_up(self):
        # The last block is full and the top block empty: a ★0 game is omitted,
        # as the plain allocator omits a game when the band is full.
        existing = {"p0": 406, "p1": 407, "p2": 408}
        markers = list(existing) + ["low"]
        got = allocate_compact_blocks(markers, {m: 2 for m in markers}, 400, 9, 3, existing)
        assert "low" not in got
        assert got == existing

    def test_reserved_numbers_are_skipped_inside_a_block(self):
        got = allocate_compact_blocks(
            ["a", "b", "c"], {"a": 1, "b": 1, "c": 1}, 400, 100, 10,
            reserved={410, 411, 413},
        )
        # Above the highest number in use in the block (413), the plain policy.
        assert got == {"a": 414, "b": 415, "c": 416}

    def test_a_block_full_of_reserved_numbers_spills(self):
        got = allocate_compact_blocks(
            ["a"], {"a": 0}, 400, 100, 10, reserved=set(range(400, 410)),
        )
        assert got == {"a": 410}

    def test_inside_a_block_the_plain_policy_allocates_above_the_highest(self):
        # 410 and 412 published, 411 freed: a new game takes 413, not the hole.
        existing = {"a": 410, "c": 412}
        got = allocate_compact_blocks(
            ["a", "c", "new"], {"a": 1, "c": 1, "new": 1}, 400, 100, 10, existing,
        )
        assert got["new"] == 413

    def test_the_block_policy_wraps_inside_its_own_block(self):
        # 417-419 taken: the top of block 1 is spent, so the hole at 410 is used,
        # not block 2.
        existing = {"x": 417, "y": 418, "z": 419}
        got = allocate_compact_blocks(
            ["x", "y", "z", "new"], {m: 1 for m in ("x", "y", "z", "new")},
            400, 100, 10, existing,
        )
        assert got["new"] == 410

    def test_published_games_hold_whatever_block_they_are_in(self):
        # 'a' scored ★9 when published on 455 (the ★4 block); now it belongs in
        # block 0, and it stays on 455.
        got = allocate_compact_blocks(["a"], {"a": 0}, 400, 100, 10, {"a": 455})
        assert got == {"a": 455}

    def test_a_band_narrower_than_every_block_clamps_into_its_last(self):
        # 25 numbers, blocks of 10: blocks 0, 1 and a short 420-424; a ★5 game
        # (block 4) still gets a number, in the last block, not dropped.
        got = allocate_compact_blocks(["mid", "top"], {"mid": 4, "top": 0}, 400, 25, 10)
        assert got == {"top": 400, "mid": 420}

    def test_an_unknown_marker_goes_to_the_last_block(self):
        got = allocate_compact_blocks(["a"], {}, 400, 100, 10)
        assert got == {"a": 490}

    def test_empty_band_is_empty(self):
        assert allocate_compact_blocks(["a"], {"a": 0}, 400, 0, 10) == {}

    def test_never_a_duplicate_number(self):
        rng = random.Random(7)
        for _ in range(200):
            markers = [f"m{i}" for i in range(rng.randint(0, 60))]
            blocks = {m: rng.randint(0, 9) for m in markers}
            existing = {m: rng.randint(380, 520) for m in markers if rng.random() < 0.4}
            reserved = {rng.randint(400, 499) for _ in range(rng.randint(0, 15))}
            got = allocate_compact_blocks(markers, blocks, 400, 100, 10, existing, reserved)
            assert len(set(got.values())) == len(got)
            assert all(400 <= n <= 499 for n in got.values())
            assert not (set(got.values()) & reserved)

    def test_one_block_covering_the_band_is_the_plain_allocator(self):
        """Pass 1 is shared and each block IS allocate_compact_numbers, so a
        single block must reproduce it exactly."""
        rng = random.Random(11)
        for _ in range(200):
            markers = [f"m{i}" for i in range(rng.randint(0, 40))]
            existing = {m: rng.randint(395, 430) for m in markers if rng.random() < 0.5}
            reserved = {rng.randint(400, 424) for _ in range(rng.randint(0, 5))}
            size = rng.randint(1, 30)
            plain = allocate_compact_numbers(markers, 400, size, existing, reserved)
            one = allocate_compact_blocks(
                markers, {m: 0 for m in markers}, 400, size, size, existing, reserved,
            )
            assert one == plain


# --------------------------------------------------- plugin-level helpers ---

@pytest.mark.parametrize("score, block", [
    (10.0, 0), (9.99, 0), (9.0, 0), (8.96, 0),  # 8.96 is named ★9.0
    (8.94, 1), (8.0, 1), (7.5, 2), (1.0, 8), (0.99, 8),  # 0.99 is named ★1.0
    (0.94, 9), (0.4, 9), (0.0, 9), (None, 9), (-2.0, 9),
    (float("nan"), 9), ("junk", 9), (11.5, 0),
])
def test_score_block_follows_the_displayed_whole_star(plugin, score, block):
    assert plugin._score_block(score) == block


@pytest.mark.parametrize("settings, size", [
    ({}, None),
    ({"compact_score_buckets": False, "compact_bucket_size": 5}, None),
    ({"compact_score_buckets": True}, 10),
    ({"compact_score_buckets": True, "compact_bucket_size": 5}, 5),
    ({"compact_score_buckets": True, "compact_bucket_size": "20"}, 20),
    ({"compact_score_buckets": True, "compact_bucket_size": 0}, 10),
    ({"compact_score_buckets": True, "compact_bucket_size": -4}, 10),
    ({"compact_score_buckets": True, "compact_bucket_size": "abc"}, 10),
    ({"compact_score_buckets": True, "compact_bucket_size": None}, 10),
])
def test_bucket_size_resolution(plugin, settings, size):
    assert plugin._resolve_compact_bucket_size(settings) == size


def _game(i, score, minute=None, home=None):
    return {
        "sport_prefix": "TEST",
        "away": f"Away{i}",
        "home": home or f"Home{i}",
        "start_time_utc": f"2026-08-29T12:{(i if minute is None else minute):02d}:00+00:00",
        "extra": {"espn_id": f"e{i}"},
        "score": score,
    }


def _assign(plugin, games, **kw):
    kw.setdefault("mode", "compact")
    kw.setdefault("band_size", 100)
    return plugin._assign_channel_numbers(games, 400, timezone.utc, **kw)


def _by_index(plugin, games, got):
    return {i: got.get(plugin._build_marker_key(g)) for i, g in enumerate(games)}


class TestAssignChannelNumbers:
    def test_games_land_in_their_score_blocks_in_kickoff_order(self, plugin):
        games = [_game(0, 5.2), _game(1, 9.4), _game(2, 10.0), _game(3, 0.3), _game(4, 5.9)]
        got = _by_index(plugin, games, _assign(plugin, games, bucket_size=10))
        assert got == {1: 400, 2: 401, 0: 440, 4: 441, 3: 490}

    def test_off_is_exactly_the_plain_allocator(self, plugin):
        games = [_game(i, s) for i, s in enumerate([5.2, 9.4, 10.0, 0.3, 5.9])]
        plain = _assign(plugin, games)
        assert _assign(plugin, games, bucket_size=None) == plain
        assert sorted(plain.values()) == [400, 401, 402, 403, 404]

    def test_kickoff_mode_ignores_it(self, plugin):
        games = [_game(i, s) for i, s in enumerate([5.2, 9.4, 10.0, 0.3, 5.9])]
        plain = plugin._assign_channel_numbers(games, 400, timezone.utc)
        bucketed = plugin._assign_channel_numbers(games, 400, timezone.utc, bucket_size=10)
        assert bucketed == plain

    def test_a_published_game_keeps_its_number_when_its_score_changes(self, plugin):
        games = [_game(0, 9.5), _game(1, 3.1)]
        first = _assign(plugin, games, bucket_size=10)
        assert sorted(first.values()) == [400, 460]
        games[0]["score"], games[1]["score"] = 2.0, 9.9
        again = _assign(plugin, games, bucket_size=10, existing_numbers=first)
        assert again == first

    def test_reserved_numbers_are_skipped_inside_a_block(self, plugin):
        games = [_game(0, 8.1), _game(1, 8.7)]
        got = _by_index(plugin, games, _assign(
            plugin, games, bucket_size=10, reserved_numbers={410, 411}))
        assert got == {0: 412, 1: 413}

    def test_favorites_take_a_block_of_their_own_ahead_of_the_scores(self, plugin):
        games = [_game(0, 9.8), _game(1, 1.2, home="Georgia"), _game(2, 5.0)]
        leaders = frozenset({plugin._build_marker_key(games[1])})
        got = _by_index(plugin, games, _assign(
            plugin, games, bucket_size=10, band_size=110, leaders=leaders))
        assert got == {1: 400, 0: 410, 2: 450}

    def test_favorites_block_exists_with_no_favorite(self, plugin):
        """On with an empty favorites list: the score blocks still start one
        block down, so listing a favorite later moves nobody."""
        games = [_game(0, 9.8)]
        got = _assign(plugin, games, bucket_size=10, band_size=110, leaders=frozenset())
        assert list(got.values()) == [410]

    def test_favorites_spill_down_into_the_score_blocks(self, plugin):
        games = [_game(i, 4.0, home="Georgia") for i in range(3)] + [_game(9, 9.0)]
        leaders = frozenset(plugin._build_marker_key(g) for g in games[:3])
        got = _by_index(plugin, games, _assign(
            plugin, games, bucket_size=2, band_size=22, leaders=leaders))
        assert got == {0: 400, 1: 401, 2: 402, 3: 403}

    def test_a_narrow_band_warns_and_still_places_every_game(self, plugin, caplog):
        games = [_game(i, s) for i, s in enumerate([9.0, 5.0, 1.0])]
        with caplog.at_level("WARNING"):
            got = _by_index(plugin, games, _assign(plugin, games, bucket_size=10, band_size=25))
        assert got == {0: 400, 1: 420, 2: 421}
        assert "Group by score" in caplog.text and "at least 100" in caplog.text

    def test_a_full_band_drops_the_lowest_scores(self, plugin, caplog):
        # Blocks of one: 400 = ★9-10, 401 = ★8, 402 = everything lower (the
        # band's last block). The second ★8 game spills into 402 ahead of the
        # ★0 game, which is the one left out.
        games = [_game(i, s) for i, s in enumerate([0.5, 9.0, 8.0, 8.2])]
        with caplog.at_level("WARNING"):
            got = _by_index(plugin, games, _assign(plugin, games, bucket_size=1, band_size=3))
        assert got == {1: 400, 2: 401, 3: 402, 0: None}
        assert "lowest-scored" in caplog.text


# ------------------------------------------------------------- manifest ---

def _field(fid):
    with open(os.path.join(HERE, os.pardir, "plugin.json"), encoding="utf-8") as f:
        return next(x for x in json.load(f)["fields"] if x["id"] == fid)


def test_the_settings_are_declared_off_and_default_ten(plugin):
    on = _field("compact_score_buckets")
    size = _field("compact_bucket_size")
    assert on["type"] == "boolean" and on["default"] is False
    assert size["type"] == "number" and size["default"] == plugin.DEFAULT_COMPACT_BUCKET_SIZE == 10


def test_the_help_text_says_a_published_game_never_moves():
    text = _field("compact_score_buckets")["help_text"].lower()
    assert "first published" in text and "never moves" in text
    assert "original block" in text
    assert "kickoff order" in text
    assert "gaps" in text
    assert "compact range size" in text and "favorites first" in text


# ------------------------------------------------- scenarios (subprocess) ---

def _load_kit():
    spec = importlib.util.spec_from_file_location(
        "_apply_harness_kit", os.path.join(HERE, "test_apply_empty_slate.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_kit = _load_kit()
_settings, _seed_source_channel, _row, _write_cache, _publish, _owned = (
    _kit._settings, _kit._seed_source_channel, _kit._row,
    _kit._write_cache, _kit._publish, _kit._owned,
)

KICKOFF = datetime(2030, 1, 5, 20, 0, tzinfo=timezone.utc)
BUCKETS = dict(channel_numbering_mode="compact", virtual_channel_base=400,
               compact_band_size=110, compact_score_buckets=True)


def _scored(gid, start, channel_id, score, home="Texas"):
    row = _row(gid, start, channel_id, home=home, away="Oklahoma")
    row["score"] = score
    return row


def _marker(gid):
    return f"ranked_matchups:CFB:{gid}"


def _numbers(env):
    return {m: int(c.channel_number) for m, c in _owned(env).items()}


def _apply(env, games, **over):
    _write_cache(env, games)
    result = env.plugin._action_apply(_settings(**over))
    assert result["status"] == "ok", result
    return result


def scenario_apply_places_new_games_by_score(env):
    user_ch, _ = _seed_source_channel(env)
    games = [_scored(1, KICKOFF, user_ch.id, 6.4), _scored(2, KICKOFF + timedelta(hours=1), user_ch.id, 9.1),
             _scored(3, KICKOFF + timedelta(hours=2), user_ch.id, 6.9)]
    _publish(env, games, settings=_settings(**BUCKETS))
    assert _numbers(env) == {_marker(2): 400, _marker(1): 430, _marker(3): 431}


def scenario_a_score_change_never_moves_a_published_game(env):
    user_ch, _ = _seed_source_channel(env)
    games = [_scored(1, KICKOFF, user_ch.id, 6.4), _scored(2, KICKOFF + timedelta(hours=1), user_ch.id, 9.1)]
    _publish(env, games, settings=_settings(**BUCKETS))
    before = _numbers(env)
    games[0]["score"], games[1]["score"] = 9.7, 2.2
    newcomer = _scored(4, KICKOFF + timedelta(hours=3), user_ch.id, 9.3)
    _apply(env, games + [newcomer], **BUCKETS)
    after = _numbers(env)
    assert {m: after[m] for m in before} == before
    # The newcomer goes above the highest number used in the ★9-10 block.
    assert after[_marker(4)] == 401


def scenario_favorites_first_puts_favorites_ahead_of_the_blocks(env):
    user_ch, _ = _seed_source_channel(env)
    games = [_scored(1, KICKOFF, user_ch.id, 9.5), _scored(2, KICKOFF + timedelta(hours=1), user_ch.id, 3.0,
                                                          home="Georgia")]
    _publish(env, games, settings=_settings(favorites_first=True, favorites="Georgia", **BUCKETS))
    assert _numbers(env) == {_marker(2): 400, _marker(1): 410}


def scenario_kickoff_mode_ignores_the_setting(env):
    user_ch, _ = _seed_source_channel(env)
    games = [_scored(1, KICKOFF, user_ch.id, 6.4), _scored(2, KICKOFF + timedelta(hours=1), user_ch.id, 9.1)]
    _publish(env, games, settings=_settings(compact_score_buckets=True))
    floor = 11  # the auto floor above the user's own channel 10
    assert _numbers(env) == {
        _marker(g): env.plugin.stable_channel_number(floor, start, _marker(g), timezone.utc)
        for g, start in ((1, KICKOFF), (2, KICKOFF + timedelta(hours=1)))
    }


def scenario_setting_off_is_plain_compact(env):
    user_ch, _ = _seed_source_channel(env)
    games = [_scored(1, KICKOFF, user_ch.id, 6.4), _scored(2, KICKOFF + timedelta(hours=1), user_ch.id, 9.1)]
    _publish(env, games, settings=_settings(
        channel_numbering_mode="compact", virtual_channel_base=400, compact_band_size=110,
        compact_bucket_size=5))
    assert _numbers(env) == {_marker(1): 400, _marker(2): 401}


SCENARIOS = _kit.collect_scenarios(globals())


@pytest.fixture(scope="module")
def harness_results(tmp_path_factory):
    return _kit.run_harness(tmp_path_factory, __file__, SCENARIOS)


@pytest.mark.parametrize("name", SCENARIOS)
def test_compact_score_buckets(harness_results, name):
    error = harness_results.get(name, "scenario did not run")
    assert error is None, error
