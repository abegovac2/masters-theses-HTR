from typing import List
from groq import AsyncGroq, RateLimitError
from functools import cache
import asyncio


class GroqClient:

    def __init__(
        self, groq_model: str, groq_api_keys: List[str], few_shot_size: int
    ) -> None:
        self._current_key = 0
        self._groq_api_keys = groq_api_keys
        self._client = AsyncGroq(api_key=self._groq_api_keys[self._current_key])
        self._groq_model = groq_model
        self._few_shot_size = few_shot_size
        self._semaphore = asyncio.Semaphore(15)

    def _cycle_api_key(self):
        self._current_key = (self._current_key + 1) % len(self._groq_api_keys)
        self._client = AsyncGroq(api_key=self._groq_api_keys[self._current_key])

    @cache
    def get_few_shot(self):
        few_shot = ""
        with open("llm_ds_faulty.csv", "r", encoding="utf8") as rf:
            for i in range(self._few_shot_size):
                line = rf.readline()
                line = line.split(",")
                few_shot += f"{line[0]} => {line[1]}"
        return few_shot

    def get_messages(self, user_prompt: str) -> str:

        return [
            {
                "role": "system",
                "content": f"""
                    Tvoj zadatak je da prepraviš rečenicu tako da ima smisla.
                    Rečenica je iz minskog zapisnika i može sadržavati greške u kucanju ili druge nepravilnosti.
                    Tvoj odgovor treba da bude samo ispravljena rečenica,
                    bez dodavanja dodatnih komentara ili objašnjenja.

                    Evo nekoliko primjera kako ta prepravka treba da izgleda,
                    na lijevoj strani su pogrešne rečenice a na desnoj njihove prepravke:
                    {self.get_few_shot()}

                    U svome odgovoru samo vraćaj preravke.
                    """,
            },
            {"role": "user", "content": f"{user_prompt} => "},
        ]

    async def correct_extraction(self, orginal_txt: str, call_num: int = 0):
        if not orginal_txt:
            return orginal_txt

        if call_num > 3:
            print("To many requests made, returning unaltered data")
            return orginal_txt
        messages = self.get_messages(orginal_txt)

        try:
            async with self._semaphore:
                response = await self._client.chat.completions.create(
                    messages=messages, model=self._groq_model
                )
                return response.choices[0].message.content.upper()
        except RateLimitError as e:
            retry_after = float(e.response.headers.get("retry-after", 0))
            print(f"Exception ", e)
            print(f"Retrying after {retry_after}s")
            # self._cycle_api_key()
            await asyncio.sleep(float(retry_after))
            # call_num += 1
            return await self.correct_extraction(orginal_txt, call_num)
