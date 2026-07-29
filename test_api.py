import os

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    """Run an explicit, billable OpenAI connectivity smoke test."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes text.",
            },
            {
                "role": "user",
                "content": (
                    "Summarize what the EPLC Development Phase focuses on "
                    "in one sentence."
                ),
            },
        ],
    )
    print("Model Output:")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
