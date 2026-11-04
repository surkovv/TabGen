import importlib.util
import os
import unittest

from tabgen.analyze_act import OpenAILLMClient


def _openai_installed() -> bool:
    return importlib.util.find_spec("openai") is not None


class OpenAILiveTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("RUN_OPENAI_LIVE_TEST") == "1",
        "set RUN_OPENAI_LIVE_TEST=1 to run the live OpenAI integration test",
    )
    @unittest.skipUnless(os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY is not set")
    @unittest.skipUnless(_openai_installed(), "openai package is not installed")
    def test_openai_llm_client_live(self):
        model = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
        response_kwargs = {
            "max_output_tokens": int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "2048")),
        }
        if model.startswith(("gpt-5", "o")):
            response_kwargs["reasoning"] = {
                "effort": os.environ.get("OPENAI_REASONING_EFFORT", "low"),
            }
        llm = OpenAILLMClient(model=model, **response_kwargs)
        text = llm.generate(
            [
                {
                    "role": "user",
                    "content": "Return exactly: tabgen-openai-client-ok",
                }
            ]
        )

        self.assertIn("tabgen-openai-client-ok", text)


if __name__ == "__main__":
    unittest.main()
