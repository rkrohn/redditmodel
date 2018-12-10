
import file_utils
import operator


#first question: how many big graphs are there?

#load all three domain breakdown files
crypto_subreddit_dist = file_utils.load_json("results/crypto_post_subreddit_dist.json")
cve_subreddit_dist = file_utils.load_json("results/cve_post_subreddit_dist.json")
cyber_subreddit_dist = file_utils.load_json("results/cyber_post_subreddit_dist.json")

subreddit_dict = {key: value for key, value in cve_subreddit_dist.items()}		#cve
subreddit_dict.update(crypto_subreddit_dist)
subreddit_dict.update(cyber_subreddit_dist)
print(subreddit_dict)

sorted_subs = sorted(subreddit_dict.items(), key=operator.itemgetter(1))
for pair in sorted_subs:
	print(pair[0], pair[1])

#well, here's the biggest - sorted
'''
Lisk 2537
ReverseEngineering 2614
netsecstudents 3149
SocialEngineering 4258			
linuxadmin 4534
crypto 4666						<<< this is as big as I've managed to run node2vec on
antivirus 5921
Malware 6856
msp 7735
AskNetsec 8460
compsci 9657
HowToHack 12123
Monero 12941
netsec 17697
talesfromtechsupport 21907
security 22689					<--- and so did this one
hacking 28894					<--- and this one
networking 29504				<--- and this one
privacy 30559					<--- and this one
softwaregore 35858				<--- and this one (~2.5 hours)
Piracy 40859					<--- and this one (~3 hours)
linux 44053						<---and this one (~2.5 hours to fit and build graph)
ethereum 50403					<--- this graph did build (eventually)
programming 88311
sysadmin 96119
Windows10 115529
Android 212623
Bitcoin 232633
techsupport 367698
technology 407695
pcmasterrace 781999

'''
#ethereum built fine, and graph is 'only' 7GB
#how long does that take to node2vec? running now
#KILLED

#security also built (second-largest so far), and graph file is just under 1GB
#time node2vec on that as well
#KILLED

#and time compsci (noticeably smaller)
#graph file is 0.15GB - much more reasonable
#DIED - assume graph too big

#try Malware node2vec - 6856 posts -> 6548 posts -> 5385541 edges
#DIED - but I did start up another one at the same time...

#and SocialEngineering node2vec - 4258 posts -> 3437 posts -> 1956406 edges
#(start again since I mucked it up)
#finished, under 5 minutes

#try antivirus - 5921 posts -> 5667 posts -> 6133015 edges
#DIED

#try crypto - 4666 posts -> 3815 posts -> ~155000 edges
#success! under 5 minutes

#try antivirus again
#nope, totally dead - guess that's the limit


#how much info will we keep if we have to sample down the big graphs to these levels?
#					50K		30K		20K		10K
#	Bitcoin			21%		13%		8.5%	4.2%
#	pcmasterrace	6.4%	3.8%	2.5%	1.2%
#not ideal


#how many users in some of these big graphs?
'''
print("\nUnique active users:")
users_filepath = "model_files/users/%s_users.txt"
user_ids = file_utils.load_pickle(users_filepath % "pcmasterrace")
print(len(set(user_ids)), "pcmasterrace")
user_ids = file_utils.load_pickle(users_filepath % "Bitcoin")
print(len(set(user_ids)), "Bitcoin")
user_ids = file_utils.load_pickle(users_filepath % "Windows10")
print(len(set(user_ids)), "Windows10")
'''

'''
Unique active users:
260347 pcmasterrace
133975 Bitcoin
79223 Windows10
'''

#so we probably can't even keep all the unique users!
#wait, that was commenting users, not just posting

#how many unique posting users? and tokens?

'''
print("\nUnique posting users and post tokens:")
posts_filepath = "model_files/posts/%s_posts.pkl"

posts = file_utils.load_pickle(posts_filepath % "pcmasterrace")
print("pcmasterrace", ":")
print("  ", len(set([post['user'] for postid, post in posts.items()])), "users")
print("  ", len(set([token for postid, post in posts.items() for token in post['tokens']])), "tokens")

posts = file_utils.load_pickle(posts_filepath % "Bitcoin")
print("Bitcoin", ":")
print("  ", len(set([post['user'] for postid, post in posts.items()])), "users")
print("  ", len(set([token for postid, post in posts.items() for token in post['tokens']])), "tokens")

posts = file_utils.load_pickle(posts_filepath % "Windows10")
print("Windows10", ":")
print("  ", len(set([post['user'] for postid, post in posts.items()])), "users")
print("  ", len(set([token for postid, post in posts.items() for token in post['tokens']])), "tokens")
'''

'''
Unique posting users and post tokens:
pcmasterrace :
   162071 users
   103639 tokens
Bitcoin :
   56643 users
   84977 tokens
Windows10 :
   55064 users
   29701 tokens
'''

#nope, can't please everybody - or keep all the users
#this sort of points to a dynamic graph definition, so we can keep the most relevant information
#ie, keep users and tokens of seed posts, and not much else
#craaaaaaap

#looks like sysadmin done - verify
'''
posts_filepath = "model_files/posts/%s_posts.pkl"
posts = file_utils.load_pickle(posts_filepath % "sysadmin")
print(len(posts), "in sysadmin")
'''
#yep - post and params counts match, even if 70127 is less than the 96119 expected

'''
#check Win10
posts_filepath = "model_files/posts/%s_posts.pkl"
posts = file_utils.load_pickle(posts_filepath % "Windows10")
print(len(posts), "in Windows10")
'''
#yep again: 102475 vs 115529 expected, but consistent across params and posts

#and techsupport
posts_filepath = "model_files/posts/%s_posts.pkl"
posts = file_utils.load_pickle(posts_filepath % "techsupport")
print(len(posts), "in techsupport")

#yep - 332424 posts and params (down from 367698 expected)