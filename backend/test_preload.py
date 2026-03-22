import asyncio
import ollama

async def main():
    client = ollama.AsyncClient()
    try:
        await client.chat(model="gemma3:4b", messages=[], keep_alive=-1)
        print("loaded")
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(main())
