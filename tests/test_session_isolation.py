"""Exercise production functions with fake providers, without loading the UI."""

import ast
import os
import types
import unittest
import uuid
from collections import OrderedDict
from pathlib import Path


class Provider:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @classmethod
    def from_documents(cls, *args, **kwargs):
        cls.calls.append((args, kwargs))
        return cls()

    def as_retriever(self, **kwargs):
        return self


class SessionIsolationTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).parents[1] / "delta-chatbot" / "app.py"
        names = {
            "is_api_key_valid",
            "create_vector_store_retriever",
            "create_llm",
            "clean_chain",
        }
        definitions = [
            node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        self.state = types.SimpleNamespace(
            chain="old", messages=OrderedDict(old="answer")
        )
        self.ns = {
            "os": os,
            "uuid": uuid,
            "OrderedDict": OrderedDict,
            "st": types.SimpleNamespace(
                session_state=self.state,
                sidebar=types.SimpleNamespace(warning=lambda *a, **kw: None),
            ),
            "Chroma": Provider,
            "OpenAIEmbeddings": Provider,
            "HuggingFaceHubEmbeddings": Provider,
            "ChatOpenAI": Provider,
            "HuggingFaceHub": Provider,
        }
        exec(
            compile(
                ast.Module(body=definitions, type_ignores=[]),
                str(source),
                "exec",
            ),
            self.ns,
        )
        Provider.calls = []

    def test_validation_never_changes_process_credentials(self):
        before = dict(os.environ)
        self.assertTrue(self.ns["is_api_key_valid"]("openai", "sk-test"))
        self.assertTrue(self.ns["is_api_key_valid"]("huggingface", "hf_test"))
        self.assertFalse(self.ns["is_api_key_valid"]("unknown", "test"))
        self.assertEqual(dict(os.environ), before)

    def test_each_document_set_uses_a_separate_collection(self):
        build = self.ns["create_vector_store_retriever"]
        build("openai", ["document-a"], "sk-user-a")
        build("openai", ["document-b"], "sk-user-b")
        first, second = Provider.calls
        self.assertNotEqual(
            first[1]["collection_name"], second[1]["collection_name"]
        )
        self.assertEqual(first[0][1].kwargs["openai_api_key"], "sk-user-a")
        self.assertEqual(second[0][1].kwargs["openai_api_key"], "sk-user-b")

    def test_llm_credentials_are_bound_to_each_client(self):
        build = self.ns["create_llm"]
        self.assertEqual(
            build("openai/test", "sk-a").kwargs["openai_api_key"], "sk-a"
        )
        self.assertEqual(
            build("org/test", "hf_b").kwargs["huggingfacehub_api_token"],
            "hf_b",
        )

    def test_reset_clears_chain_and_displayed_history(self):
        self.ns["clean_chain"]()
        self.assertIsNone(self.state.chain)
        self.assertEqual(self.state.messages, OrderedDict())


if __name__ == "__main__":
    unittest.main()
