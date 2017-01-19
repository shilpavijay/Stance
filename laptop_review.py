import opinion_mining as om
from nltk.tokenize import sent_tokenize

Review1 = "It is almost best product. When u install windows 7 u gonna face drivers problem, best option to install windows 10 on it. Overall product is good for use.Sound quality average."
Review2 = "Defective product i got delivered it on 28 oct 2016 in eve but it did not start and when complained to flipkart on 30 oct 2016 they say problem resolved while nothing was done by them. I didnot expected this from flipkart. I am highly dissappointed with acer and flipkart.plz dont go with this product its useless. No service engineer visited to our place how could you say matter disolved you people are fooling us"
Review3 = "Very Good Product. Excellent make, easy to use. Service is really good. I would recomment to buy this product."
Review4 = "Not too good it runs very slow i am sorry but this price is good for this lappy i bought it in16k and it works perfectly......just too slow not for fast users.....or heavy users....."
# print(om.opinion(Review1))
# print(om.opinion(Review2))
# print(om.opinion(Review3))
# print(om.opinion(Review4))

def classify_review():
	all_rev = [Review1 ,Review2, Review3, Review4]
	rev_dict = {}
	for each in all_rev:
		rev_dict[each] = om.opinion(each)[0]

	print("\nPros: \n")
	for d in rev_dict:		
		if rev_dict[d] == 'Pros':
			print(d,'\n')

	print("--------------------------------------------------------------------------------------------------------------")
	print("\nCons: \n")
	for d in rev_dict:		
		if rev_dict[d] == 'Cons':
			print(d,'\n')
			
	(op,confidence) = om.opinion(" ".join(all_rev))
	print("--------------------------------------------------------------------------------------------------------------")
	print('\n\nOverall Review of this Product has more: ', op)
	print('How confident are we with the result: ', confidence.__round__(),'%')	

classify_review()
