
'''
Workflow
1.  DL data from Clarity -- 01_get_join_from_clarity.py
    a.  First, select population
        - All encounters that have the correct diagnoses, 2017 and after
        - All patients associated with those encounters
        - All encounters from those patients
        - All patients with at least 2 outpatient visits in the past 12 months
        - All notes from those outpatient visits
        - All of their diagnoses
    b.  Send some initial high and low probability notes to team
2.  Summary statistics
3.  Process text -- prep it for webanno
4.  Ship it to webanno
5.  Extract annotated data from webanno
6.  Build classifier

Known issues:  Some of my MRNs are bad.  THis is resulting in me getting too few patients.

'''



