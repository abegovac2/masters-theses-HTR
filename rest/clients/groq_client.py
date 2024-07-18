from typing import List
from groq import AsyncGroq, RateLimitError
from functools import cache
import logging


class GroqClient:

    def __init__(
        self, groq_model: str, groq_api_keys: List[str], fuel_shot_size: int
    ) -> None:
        self._current_key = 0
        self._groq_api_keys = groq_api_keys
        self._client = AsyncGroq(api_key=self._groq_api_keys[self._current_key])
        self._groq_model = groq_model
        self._fuel_shot_size = fuel_shot_size

    def _cycle_api_key(self):
        self._current_key = (self._current_key + 1) % len(self._groq_api_keys)
        self._client = AsyncGroq(api_key=self._groq_api_keys[self._current_key])

    @cache
    def get_fuel_shot(self):
        fuel_shot = ""
        with open("llm_ds_faulty.csv", "r", encoding="utf8") as rf:
            for i in range(self._fuel_shot_size):
                fuel_shot += rf.readline()
        return fuel_shot

    def get_messages(self, user_prompt: str) -> str:

        return [
            {
                "role": "system",
                "content": f"""
                    Tvoj zadatak je da prepraviš rečenicu tako da ima smisla.
                    Rečenica je iz minskog zapisnika i može sadržavati greške u kucanju ili druge nepravilnosti.
                    Tvoj odgovor treba da bude samo ispravljena rečenica,
                    bez dodavanja dodatnih komentara ili objašnjenja.

                    Evo nekoliko primjera kako to treba da izgleda:
                    {self.get_fuel_shot()}""",
            },
            {"role": "user", "content": user_prompt},
        ]

    async def correct_extraction(self, orginal_txt: str, call_num: int = 0):

        if call_num > 5:
            print("To many requests made returning unaltered data")
            return orginal_txt
        messages = self.get_messages(orginal_txt)

        try:
            response = await self._client.chat.completions.create(
                messages=messages, model=self._groq_model
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            self._cycle_api_key()
            return await self.correct_extraction(orginal_txt, call_num + 1)
