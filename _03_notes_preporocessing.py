
'''
This script duplicates much of what was in the previous script, but with a few additions:
- joining multi-word expressions
- outputting "windowed" notes.  Algo:
    initilialize empty list of MRNs who won't be eligible for the next 6 months
    for each month
        define eligible notes as people who haven't had an eligible note in the last 6 months but are otherwise eligible
        add those people to the 6-month list, along with the month in which they were added
        take those notes, and concatenate them with the last 6 months of notes
        process the combined note, and output it


'''

