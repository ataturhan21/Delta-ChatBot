"""Offline checks for document preparation and the source panel."""

import ast
import html
import os
import tempfile
import types
import unittest
from pathlib import Path


def load_functions(names, namespace):
    source = Path(__file__).parents[1] / "delta-chatbot" / "app.py"
    definitions = [
        node
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=definitions, type_ignores=[])
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


class FakeLoader:
    def __init__(self, path, **kwargs):
        self.path = path

    def load(self):
        with open(self.path, encoding="utf-8") as file:
            text = file.read()
        return [types.SimpleNamespace(page_content=text)] if text else []


class FakeSplitter:
    def __init__(self, **kwargs):
        pass

    def split_documents(self, documents):
        return list(documents)


class Upload:
    def __init__(self, name, text):
        self.name = name
        self._data = text.encode("utf-8")

    def getvalue(self):
        return self._data


class DocumentTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_functions(
            {"prepare_documents", "format_source_documents"},
            {
                "html": html,
                "os": os,
                "tempfile": tempfile,
                "PyPDFLoader": FakeLoader,
                "TextLoader": FakeLoader,
                "CharacterTextSplitter": FakeSplitter,
            },
        )

    def test_text_uploads_are_loaded(self):
        chunks = self.ns["prepare_documents"]([Upload("notes.txt", "hello")])
        self.assertEqual([c.page_content for c in chunks], ["hello"])

    def test_uploads_without_text_raise_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "No text could be extracted"):
            self.ns["prepare_documents"]([Upload("empty.txt", "")])

    def test_unsupported_formats_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported document"):
            self.ns["prepare_documents"]([Upload("notes.docx", "x")])

    def test_source_passages_are_shown_literally(self):
        sources = [
            types.SimpleNamespace(page_content="<b>bold</b> & more"),
            types.SimpleNamespace(page_content="plain"),
        ]
        rendered = self.ns["format_source_documents"](sources)
        self.assertEqual(
            rendered,
            "1 - &lt;b&gt;bold&lt;/b&gt; &amp; more<hr>2 - plain<hr>",
        )


if __name__ == "__main__":
    unittest.main()
