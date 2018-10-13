#utility methods for dealing with data

from collections import defaultdict
from operator import itemgetter
import re
import copy

#add a field and associated value to all items in list of dictionaries
def add_field(data, key, value):
	for item in data:
		item.update( {key: copy.deepcopy(value)})
	return data
#end add_field

#given a list of dictionary objects, convert to a dictionary using key_field (string) value as key
#obj[key_field] -> obj
def list_to_dict(data_list, key_field):
	data_dict = {}
	for item in data_list:
		#print a message if duplicate items to alert the user
		if item[key_field] in data_dict:
			print("\nduplicate!")
		data_dict[item[key_field]] = item 		#add item to dictionary
	return data_dict
#end list_to_dict

#given a list of dictionary objects that may contain duplicates, convert to a dictionary using key_field (string) value as key
#obj[key_field] -> list of matching objects
def list_to_dict_duplicates(data_list, key_field):
	data_dict = defaultdict(list)
	for item in data_list:
		data_dict[item[key_field]].append(item)
	return data_dict
#end list_to_dict_duplicates

#given a list of dictionary objects (all with the same set of fields), create a distribution dictionary for key_field
#dist[key_field] -> count of objects with that field value
def dictionary_field_dist(data_list, key_field):
	dist_dict = defaultdict(int)
	for key, value in data_list.items():
		dist_dict[value[key_field]] += 1
	return dist_dict
#end dictionary_field_dist

#given two list of dictionary objects (posts or comments), combine into a single list
#if duplicates (determined by id) are encountered, retain the newest version (by retrieved_on)
#as long as it does not contain any [deleted] fields
#(ie, retain as much information as possible, with a preference for the newest)
#
#identifier - indicates dictionary field used for duplicate comparisons
#time_field - dictionary field used for determining newest item (larger number newer)
#boolean_true - if False, ignore
#				if a string, set this field to True if either of duplicate objects has that field set to True
def combine_lists(data1, data2, identifier="id_h", time_field="retrieved_on", boolean_true=False):

	#convert both lists to single combined dictionary
	combined_dict = list_to_dict_duplicates(data1 + data2, identifier)

	#loop the keys, build a list with no duplicates
	combined_list = []
	for key, value_list in combined_dict.items():
		#no duplicates, take only item
		if len(value_list) == 1:
			combined_list.append(value_list[0])
		#some number of duplicates, pick one
		else:
			combined_list.append(select_item(value_list, time_field, boolean_true))

	return combined_list
#end combine_lists

#given a list of candidate items (all with matching identifiers), return the "best" single item
#take newest item that does not contain any [deleted] fields
def select_item(candidates, time_field, boolean_true=False):

	#loop items, keep track of newest without deletions
	newest_best = -1		#index of best
	newest = -1				#index of best, no field check
	#also set flag if any have boolean_true field set
	boolean_set = False
	for i in range(len(candidates)):
		#newer item, store index for time-only criteria
		if candidates[i][time_field] > candidates[newest][time_field]:
			newest = i
		#if newer and no deletions, store index for best criteria
		if candidates[i][time_field] > candidates[newest_best][time_field] and check_fields(candidates[i]) == True:
			newest_best = i
		#if boolean field set True, flag item list
		if boolean_true != False and candidates[i][boolean_true] == True:
			boolean_set = True

	selected = candidates[newest_best] if newest_best != -1 else candidates[newest]

	#set bool field true if any item was true (and this functionality was activated)
	if boolean_true != False and boolean_set == True:
		selected[boolean_true] = True
			
	return selected
#end select_item

#for a dictionary item, return true if no fields [deleted], false otherwise
def check_fields(item):
	for key, value in item:
		if isinstance(value, str) and re.search('deleted', value, re.IGNORECASE):
			return False
	return True
#enc check_fields
