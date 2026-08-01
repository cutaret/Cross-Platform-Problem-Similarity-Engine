import asyncio
from curl_cffi.requests import AsyncSession

async def test():
    async with AsyncSession(impersonate='chrome') as s:
        # Test fetching problem list via GraphQL
        query = """
        query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
            problemsetQuestionList: questionList(categorySlug: $categorySlug, limit: $limit, skip: $skip, filters: $filters) {
                total: totalNum
                questions: data {
                    questionId
                    questionFrontendId
                    title
                    titleSlug
                    difficulty
                    topicTags { name slug }
                    isPaidOnly
                }
            }
        }
        """
        r = await s.post(
            'https://leetcode.com/graphql',
            json={
                "query": query,
                "variables": {
                    "categorySlug": "all-code-essentials",
                    "skip": 0,
                    "limit": 5,
                    "filters": {}
                }
            },
            headers={"Content-Type": "application/json"}
        )
        print(r.status_code)
        data = r.json()
        pl = data['data']['problemsetQuestionList']
        print(f"Total: {pl['total']}")
        for q in pl['questions']:
            print(f"  {q['questionFrontendId']}. {q['title']} ({q['difficulty']}) paid={q['isPaidOnly']}")

asyncio.run(test())
