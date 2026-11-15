from repo2txt.focus import view

PYTHON = """\
import os


class Store:
    def get(self, key):
        return self.data[key]

    def invalidate_session(self, user):
        token = self.tokens.pop(user)
        audit(token)
        return token


def unrelated():
    return 1
"""

C_LIKE = """\
int helper(int x) {
    return x + 1;
}

static void flush_cache(cache_t *c) {
    for (int i = 0; i < c->n; i++) {
        evict(c, i);
    }
}
"""


def lines_of(v):
    return {n for a, b in v.ranges for n in range(a, b + 1)}


def test_the_matching_block_is_kept_whole_and_the_rest_outlined():
    v = view(PYTHON, {"invalidate": 3.0, "session": 3.0})

    assert lines_of(v) >= set(range(8, 12))  # the whole method
    assert 4 in lines_of(v) and 5 in lines_of(v)  # the class and a sibling, as outline
    assert 6 not in lines_of(v)  # a sibling's body
    assert "def invalidate_session(self, user):\n        token = self.tokens.pop(user)" in v.text
    assert "… lines 6-7 not shown" in v.text


def test_braces_close_the_block_they_open():
    v = view(C_LIKE, {"flush": 3.0})

    assert lines_of(v) >= set(range(5, 10))
    assert 2 not in lines_of(v)
    assert v.text.rstrip().endswith("}")


def test_common_words_alone_do_not_count_as_a_match():
    v = view(PYTHON, {"return": 0.2, "self": 0.3})

    assert 9 not in lines_of(v)


def test_line_numbers_in_markers_match_the_file():
    v = view(PYTHON, {"unrelated": 3.0})

    for line in v.text.splitlines():
        if line.startswith("… lines "):
            first, last = map(int, line.split()[2].split("-"))
            assert 1 <= first <= last <= v.lines
