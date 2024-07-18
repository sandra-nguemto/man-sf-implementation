import mkwikidata
import torch
import os
import pickle



valid_comps = {
    'Apple': 'wd:Q312',
    'Amazon': 'wd:Q3884',
    'Boeing': 'wd:Q66',
    'Bank of America': 'wd:Q487907',
    'Citigroup': 'wd:Q219508',
    'Caterpillar Inc.' : 'wd:Q459965',
    'Celgene': 'wd:Q842947',
    'Cisco': 'wd:Q173395',
    'Chevron Corporation': 'wd:Q319642',
    'Dominion Energy' : 'wd:Q677464',
    'The Walt Disney Company' : 'wd:Q7414',
    'Meta Platforms' : 'wd:Q380',
    'General Electric' : 'wd:Q54173',
    'Alphabet Inc.': 'wd:Q20800404',
    'The Home Depot' : 'wd:Q864407',
    'Intel' : 'wd:Q248',
    'Johnson & Johnson' : 'wd:Q333718',
    'JPMorgan Chase' : 'wd:Q192314',
    'The Coca-Cola Company' : 'wd:Q3295867',
    'McDonald’s' : 'wd:Q38076',
    'Merck & Co.' : 'wd:Q247489',
    'Microsoft' : 'wd:Q2283',
    'Booking Holdings' : 'wd:Q18674747',
    'Pfizer' : 'wd:Q206921',
    'AT&T': 'wd:Q35476',
    'Visa Inc.' : 'wd:Q328840',
    'Verizon Communications' : 'wd:Q467752',
    'Wells Fargo' : 'wd:Q744149',
    'Walmart' : 'wd:Q483551',
    'ExxonMobil' : 'wd:Q156238'


}




class WikiGraph():
    def __init__(self, companies: dict):
        self.comp_ids = ' '.join(list(companies.values()))
        self.comp_names = list(companies.keys())
        

    ###################  Querying Wikidata ###################      
    ## Querying [wikidata.org](https://www.wikidata.org/)

    def get_query(self):
    
        query = """
        SELECT DISTINCT ?subject ?subjectLabel ?predicate ?predicateLabel ?object ?objectLabel
        WHERE {
        VALUES ?subject{ %s }
        
        ?subject ?predicate ?object .
        
        # Ensure proper labels
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }
        
        """%(self.comp_ids)
    
        result = mkwikidata.run_query(query, params={ })
        
        return result
    
    ###################  Organizing Wikidata Statements  ###################  

    def get_statements(self):
        result = self.get_query()
        if "results" in result and "bindings" in result["results"]:
            statements = result["results"]["bindings"]
            statmts = {}
            for name in self.comp_names:
                statmts[name] = []
            for statement in statements:
                subject = statement["subjectLabel"]["value"]
                object_ = statement["objectLabel"]["value"]
                statmts[subject].append(object_)       
        else:
            print("No statements found.")
        return statmts
    

    ###################  Getting 1st & 2nd Order relations  ################### 
 
    def get_relations(self):
        statmts = self.get_statements()
        relations = {}
        statmts_list = list(statmts.keys())
        for i in range(len(statmts_list)):
            for j in range(i+1, len(statmts_list)):
                if (bool( set(statmts[statmts_list[i]]) & set(statmts[statmts_list[j]]) )) and (statmts_list[i] != statmts_list[j]):
                    relations[(statmts_list[i], statmts_list[j])] = ('2nd order relation', set(statmts[statmts_list[i]]) & set(statmts[statmts_list[j]]))
                if (statmts_list[i] in set(statmts[statmts_list[j]]) ) and (statmts_list[i] != statmts_list[j]):
                    relations[(statmts_list[i], statmts_list[j])] = ('1st order relation', set(statmts[statmts_list[j]]))
    
        return relations
    

    ###################  Getting Graph as a Matrix  ################### 

    def make_wikidata_matrix(self):

        relations = self.get_relations()
        
        matrix = torch.zeros(len(self.comp_names), len(self.comp_names))
        
        for i in range(len(self.comp_names)):
            matrix[i][i] = 1
        
        for i in range(len(self.comp_names)):
            for j in range(len(self.comp_names)):
                if (self.comp_names[i], self.comp_names[j]) in relations:
                    matrix[i][j] = 1
                    matrix[j][i] = 1

        return matrix
    

graph = WikiGraph(valid_comps)
comps_wikirelations  = graph.make_wikidata_matrix()    

if not os.path.exists("./Data/raw_data/wiki_relations"):
    os.mkdir("./Data/raw_data/wiki_relations")

with open('./Data/raw_data/wiki_relations/comps_wikirelations.pkl', 'wb') as f:
    pickle.dump(comps_wikirelations, f)
