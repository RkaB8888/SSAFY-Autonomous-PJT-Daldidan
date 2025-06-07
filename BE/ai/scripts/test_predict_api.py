# scripts/test_predict_api.py

import asyncio
import aiohttp
import time
from pathlib import Path

# 테스트할 API 주소 (로컬 or 서버 주소)
URL = "https://k12e206.p.ssafy.io/predict"  # EC2 포트에 맞게 수정 가능
HEADERS = {"accept": "application/json"}
IMAGE_PATH = "sample.jpg"  # 반드시 이 경로에 테스트 이미지가 있어야 함

# 요청 수와 동시 처리 수 설정
TOTAL_REQUESTS = 100
CONCURRENCY = 10


async def send_request(session, idx):
    with open(IMAGE_PATH, "rb") as f:
        form = aiohttp.FormData()
        form.add_field("image", f, filename="sample.jpg", content_type="image/jpeg")
        try:
            async with session.post(URL, data=form) as resp:
                status = resp.status
                if status == 200:
                    return True
                else:
                    print(f"[{idx}] ❌ Status: {status}")
                    return False
        except Exception as e:
            print(f"[{idx}] ❌ Error: {e}")
            return False


async def main():
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:
        tasks = [send_request(session, i) for i in range(TOTAL_REQUESTS)]
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start_time

        success_count = sum(results)
        fail_count = TOTAL_REQUESTS - success_count
        qps = TOTAL_REQUESTS / elapsed

        print("\n✅ 테스트 결과 요약")
        print(f"총 요청 수: {TOTAL_REQUESTS}")
        print(f"성공: {success_count}, 실패: {fail_count}")
        print(f"총 소요 시간: {elapsed:.2f}초")
        print(f"초당 처리량 (QPS): {qps:.2f} req/sec")


if __name__ == "__main__":
    asyncio.run(main())
