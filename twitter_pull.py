import tweepy
import pandas as pd

# 替换为你的API密钥
API_KEY = 'Mki9kTijKljSkrfa31rMbnkeo'
API_SECRET_KEY = 'wu6XWGSVqq2hofrtypZ3a3hPGDvcoMPWxWuOoyjeyCAJJXIHot'
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAGRrxwEAAAAA9goDwZ5HcFb4mkMUaX0HmV0kQOs%3DGsRmdkBtqfq68HsJxkFyc8CFPaF5HFhzM5WDaJ1W7e13qlgWRl'

ACCESS_TOKEN = '1874294859160289280-FmkIgn15vSbrFwRWTg1UBHu7Xj4TZj'
ACCESS_TOKEN_SECRET = 'ujQY0h8INv3R5FQeUfLpnYPC6eiKzTGlO08lYUH1dAivQ'
# 创建客户端
client = tweepy.Client(bearer_token=BEARER_TOKEN,
                       consumer_key=API_KEY,
                       consumer_secret=API_SECRET_KEY,
                       access_token=ACCESS_TOKEN,
                       access_token_secret=ACCESS_TOKEN_SECRET)

# 获取用户的推文 '1349149096909668363'
username = 'POTUS'  # 注意：这里应该是用户名，不带 @ 符号

# 首先获取用户的 ID
user = client.get_user(username=username)
user_id = user.data.id

# 使用用户 ID 获取推文
response = client.get_users_tweets(id=user_id, max_results=100, tweet_fields=['created_at', 'text'])

# 打印推文内容
for tweet in response.data:
    print(tweet.text)

# 提取推文内容进行分析
tweets_data = [{'text': tweet.text, 'created_at': tweet.created_at, 'id': tweet.id} for tweet in response.data]

# 转换为DataFrame
df = pd.DataFrame(tweets_data)

# 保存到CSV文件
df.to_csv('tweets_data.csv', index=False)

