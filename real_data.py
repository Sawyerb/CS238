"""
Calculating distributions for:
Money - Votes
Votes - Polls
"""
import csv
from collections import defaultdict, Counter

def get_cands_votes(): 
    '''
    From FEC.
    Note: tere are some rows where the same candidate repeats
    with different party. 
    '''
    cands = {}
    data = './data/2014_house_election_results.csv'
    with open(data, 'rU') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            if row['FEC ID#'] in cands and row['GENERAL %'] != '':
                cands[row['FEC ID#']]['vote %'] += float(row['GENERAL %'].replace('%', ''))
                if cands[row['FEC ID#']]['party'] not in ['R', 'D'] and row['PARTY'] in ['R', 'D']: 
                    cands[row['FEC ID#']]['party'] = row['PARTY']
            elif row['FEC ID#'] != 'n/a' and row['FEC ID#'] != '' and row['GENERAL %'] != '':
                person = {}
                person['name'] = row['CANDIDATE NAME']
                person['state'] = row['STATE ABBREVIATION']
                person['district'] = row['D'].replace(' - FULL TERM', '').replace(' - UNEXPIRED TERM', '')
                party = row['PARTY']
                if '/' in party: 
                    parties = party.split('/')
                    if 'D' in parties: 
                        party = 'D'
                    elif 'R' in parties: 
                        party = 'R'
                    else: 
                        party = row['PARTY']
                if party == 'DFL': party = 'D' # MN's dem party
                person['party'] = party.strip()
                person['vote %'] = float(row['GENERAL %'].replace('%', ''))
                cands[row['FEC ID#'].strip()] = person
    return cands

def get_money(): 
    '''
    From FEC.
    ./data/cn 2.txt is less useful than expected because for H 2014 there are often multiple Dem and Rep.
    '''
    money = defaultdict(float)
    cand_campaign_finance = './data/webl14.txt'
    with open(cand_campaign_finance, 'r') as infile:
        for line in infile:
            contents = line.strip().split('|')
            money[contents[0].strip()] = float(contents[17].strip())
    return money

def get_polls(): 
    dem = {}
    rep = {}
    data = './data/2014_house_election_polls.csv'
    with open(data, 'rU') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            dem[row['CD']] = float(row['Dem'].replace('%', ''))
            rep[row['CD']] = float(row['Rep'].replace('%', ''))
    return dem, rep
def main(): 
    '''
    Output format:
    state | district | ID | name | votes | money | poll
    '''
    cands = get_cands_votes()
    money = get_money()
    dem, rep = get_polls()
    total_election_money = defaultdict(float)
    for cand in cands:
        total_election_money[cands[cand]['state'] + cands[cand]['district']] += money[cand]
    file = open('./data/2014_house_election_clean.txt', 'w')
    writer = csv.writer(file, delimiter=',')
    writer.writerow(['state', 'district', 'id', 'name', 'votes %', 'money %', 'poll %'])
    for cand in cands:
        if money[cand] != 0.0 and cands[cand]['state'] not in ['DC', 'AS', 'VI', 'GU']:
            money_percent = money[cand] * 100 / total_election_money[cands[cand]['state'] + cands[cand]['district']]
            district = cands[cand]['district']
            if district == '00':
                district = '01'
            if cands[cand]['party'] == 'D':
                writer.writerow([cands[cand]['state'], cands[cand]['district'], cand, cands[cand]['name'], 
                    cands[cand]['vote %'], money_percent, dem[cands[cand]['state']+district]])
            if cands[cand]['party'] == 'R':
                writer.writerow([cands[cand]['state'], cands[cand]['district'], cand, cands[cand]['name'], 
                    cands[cand]['vote %'], money_percent, rep[cands[cand]['state']+district]])
    file.close()

if __name__ == '__main__':
    main()