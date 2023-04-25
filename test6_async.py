import aiofiles
import asyncio
import json
import os

convo_name_directory = r'.\conversations'

async def read():
    filename = os.path.join(convo_name_directory, 'convo.json')
    async with aiofiles.open(filename, mode='r') as f:
        contents = await f.read()
    file = json.loads(contents)
    print(f'Async reading:\n{file}\n')

async def write():
    filename = os.path.join(convo_name_directory, 'ditto_moves.txt')
    async with aiofiles.open(filename, mode='w') as f:
        await f.write('transform')
    print(f'Async writing!')

asyncio.run(read())
asyncio.run(write())