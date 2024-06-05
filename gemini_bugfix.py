from haystack.core.component.types import Variadic
from haystack import component
from typing import List, Dict, Union
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator


@component
class GeminiBugfix(GoogleAIGeminiGenerator):
    """
        Circumvent this issue: https://github.com/deepset-ai/hayhooks/issues/25
    """

    @component.output_types(replies=List[Union[str, Dict[str, str]]])
    def run(self, parts: Variadic[str]):
        parts = [p for p in parts]
        res = super(GeminiBugfix, self).run(parts)
        print(res)
        return res
