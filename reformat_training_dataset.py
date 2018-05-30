# coding: utf-8

###### Submitted for CE807 Assignment 2 by:
## Student Information
# Name: Vivan Raaj Rajalingam  Registration Number: 1704827


original = open('aij-wikiner-en-wp2.txt', 'r').readlines()

# create a new text file
wikiner = open("wikiner.txt", "w")

# initialize a new list and string variable
all_x = []
point = []
newf = ''

# loop through lines in dataset
for line in original:
    #  strip at empty space and split sentence at space
    cleaned = line.strip().split(' ')
    # loop through cleaned file
    for clean in cleaned:
        t = clean.split('|')
        if(len(t) > 1):
            del t[1]
            newf = t[0]+' '+t[1]
        wikiner.write(newf)
        wikiner.write('\n')
    wikiner.write('\n')
wikiner.close()