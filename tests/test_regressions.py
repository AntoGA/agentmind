from types import SimpleNamespace
from datetime import datetime, timedelta
import pytest
from agentmind import Memory, MemoryConfig
from agentmind.core.store import MemoryStore, MemoryRecord


@pytest.fixture
def store():
    value = MemoryStore(SimpleNamespace(storage='sqlite', db_path=':memory:'))
    yield value
    value.conn.close()


def save(store, text, group='keep', vector=None, agent='a', timestamp=None):
    return store.save(MemoryRecord(agent_id=agent, content=text,
        embedding=[1.0, 0.0] if vector is None else vector,
        metadata={'group': group}, timestamp=timestamp or datetime.now()))


def test_public_config_export():
    assert MemoryConfig(storage='sqlite').storage == 'sqlite'


def test_delete_filters_do_not_delete_other_memories(store):
    keep = save(store, 'keep')
    remove = save(store, 'remove', group='remove')
    other = save(store, 'other agent', group='remove', agent='b')
    assert store.delete('a', filters={'group': 'remove'}) == 1
    assert [m.id for m in store.search('a')] == [keep]
    assert [m.id for m in store.search('b')] == [other]


def test_empty_id_list_does_not_delete_everything(store):
    save(store, 'keep')
    assert store.delete('a', memory_ids=[]) == 0
    assert store.get_stats('a')['total_memories'] == 1


def test_delete_selectors_intersect(store):
    now = datetime.now()
    old = save(store, 'old', group='remove', timestamp=now-timedelta(days=2))
    recent = save(store, 'recent', group='remove', timestamp=now)
    keep = save(store, 'wrong group', timestamp=now-timedelta(days=2))
    assert store.delete('a', memory_ids=[old, recent, keep],
        filters={'group': 'remove'}, older_than=now-timedelta(days=1)) == 1
    assert {m.id for m in store.search('a')} == {recent, keep}


def test_recall_finds_older_matching_record(store):
    old = save(store, 'old', vector=[1, 0], timestamp=datetime.now()-timedelta(days=1))
    for i in range(12):
        save(store, str(i), group='other', vector=[0, 1])
    assert store.search('a', query_embedding=[1, 0], limit=1)[0].id == old
    assert store.search('a', filters={'group': 'keep'}, limit=1)[0].id == old


def test_zero_similarity_is_included_at_zero_threshold(store):
    save(store, 'orthogonal', vector=[0, 1])
    results = store.search('a', query_embedding=[1, 0], min_score=0)
    assert len(results) == 1 and results[0].score == 0


@pytest.mark.parametrize('vector', [[0, 0], [float('nan'), 1], [1, 2, 3]])
def test_invalid_or_incompatible_embedding_is_rejected(store, vector):
    save(store, 'record')
    with pytest.raises(ValueError):
        store.search('a', query_embedding=vector)


def test_memory_forget_forwards_filter():
    memory = Memory(MemoryConfig(db_path=':memory:'))
    try:
        memory.remember('a', 'first', metadata={'group':'keep'})
        memory.remember('a', 'second', metadata={'group':'remove'})
        assert memory.forget('a', filters={'group':'remove'}) == 1
        assert memory.recall('a')[0].content == 'first'
    finally:
        memory.store.conn.close()


def test_cleanup_considers_oldest_records_not_just_recent_window():
    memory = Memory(MemoryConfig(db_path=':memory:', max_memories=2))
    try:
        low = memory.remember('a', 'old low', importance=0.1,
            timestamp=datetime.now()-timedelta(days=10))
        high = memory.remember('a', 'new high', importance=0.9)
        medium = memory.remember('a', 'new medium', importance=0.5)
        assert {m.id for m in memory.recall('a')} == {high, medium}
    finally:
        memory.store.conn.close()
