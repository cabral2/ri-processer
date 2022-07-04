from cProfile import label
from typing import List
from abc import abstractmethod
from typing import List, Set,Mapping
from index.structure import TermOccurrence
import math
from enum import Enum

class IndexPreComputedVals():
    def __init__(self,index):
        self.index = index
        self.precompute_vals()

    def precompute_vals(self):
        """
        Inicializa os atributos por meio do indice (idx):
            doc_count: o numero de documentos que o indice possui
            document_norm: A norma por documento (cada termo é presentado pelo seu peso (tfxidf))
        """
        self.document_norm = {}
        self.doc_count = self.index.document_count

        vocabulary = self.index.vocabulary

        for term in vocabulary:
            occurrence_list = self.index.get_occurrence_list(term)
            for occurrence in occurrence_list:
                result = VectorRankingModel.tf_idf(self.doc_count, occurrence.term_freq, len(occurrence_list)) ** 2
                print(result)
                if occurrence.doc_id in self.document_norm:
                    self.document_norm[occurrence.doc_id] += result
                else:
                    self.document_norm[occurrence.doc_id] = result
        
        for id, value in self.document_norm.items():
            self.document_norm[id] = math.sqrt(value)

        print(self.document_norm)

        return self.document_norm
                
        

        
class RankingModel():
    @abstractmethod
    def get_ordered_docs(self,query:Mapping[str,TermOccurrence],
                              docs_occur_per_term:Mapping[str,List[TermOccurrence]]) -> (List[int], Mapping[int,float]):
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    def rank_document_ids(self,documents_weight):
        doc_ids = list(documents_weight.keys())
        doc_ids.sort(key= lambda x:-documents_weight[x])
        return doc_ids

class OPERATOR(Enum):
  AND = 1
  OR = 2
    
#Atividade 1
class BooleanRankingModel(RankingModel):
    def __init__(self,operator:OPERATOR):
        self.operator = operator

    def intersection_all(self,map_lst_occurrences:Mapping[str,List[TermOccurrence]]) -> List[int]:
        
        set_ids = set()
        values = list(map_lst_occurrences.values())
        if len(values) == 0:
            return []
 
        initial_ids = map(lambda x : x.doc_id , values[0])
        for value in list(initial_ids):
            set_ids.add(value)


        for  term, lst_occurrences in map_lst_occurrences.items():
            ids = map(lambda x : x.doc_id , lst_occurrences)
            set_ids = set_ids.intersection(ids)
            print(set_ids)

        return list(set_ids)

    def union_all(self,map_lst_occurrences:Mapping[str,List[TermOccurrence]]) -> List[int]:
        set_ids = set()
        
        for  term, lst_occurrences in map_lst_occurrences.items():
            ids = map(lambda x : x.doc_id , lst_occurrences)
            for value in list(ids):
                set_ids.add(value)

        return list(set_ids)

    def get_ordered_docs(self,query:Mapping[str,TermOccurrence],
                              map_lst_occurrences:Mapping[str,List[TermOccurrence]]) -> (List[int], Mapping[int,float]):
        """Considere que map_lst_occurrences possui as ocorrencias apenas dos termos que existem na consulta"""
        if self.operator == OPERATOR.AND:
            return self.intersection_all(map_lst_occurrences),None
        else:
            return self.union_all(map_lst_occurrences),None

#Atividade 2
class VectorRankingModel(RankingModel):

    def __init__(self,idx_pre_comp_vals:IndexPreComputedVals):
        self.idx_pre_comp_vals = idx_pre_comp_vals

    @staticmethod
    def tf(freq_term:int) -> float:
        return 1 + math.log(freq_term, 2)

    @staticmethod
    def idf(doc_count:int, num_docs_with_term:int )->float:
        return math.log(doc_count/num_docs_with_term, 2)

    @staticmethod
    def tf_idf(doc_count:int, freq_term:int, num_docs_with_term) -> float:
        tf = VectorRankingModel.tf(freq_term)
        idf = VectorRankingModel.idf(doc_count, num_docs_with_term)
        #print(f"TF:{tf} IDF:{idf} n_i: {num_docs_with_term} N: {doc_count}")
        return tf*idf

    def get_ordered_docs(self,query:Mapping[str,TermOccurrence],
                              docs_occur_per_term:Mapping[str,List[TermOccurrence]]) -> (List[int], Mapping[int,float]):
            documents_weight = {}

            relevant_doc_ids = {}
            for query_term, query_term_occur in query.items():
                relevant_doc_ids[query_term] = list(map(lambda x : x.doc_id, docs_occur_per_term[query_term]))


            for term, doc_ids in relevant_doc_ids.items():
                list_term_occur = docs_occur_per_term[term]
                for doc_id in doc_ids:
                    doc_value = self.idx_pre_comp_vals.document_norm[doc_id]
                    soma = 0
                    for term_occur in list_term_occur:
                        if term_occur.doc_id == doc_id:
                            wij = self.tf_idf(self.idx_pre_comp_vals.doc_count, term_occur.term_freq, len(list_term_occur))
                            wiq =  self.tf_idf(self.idx_pre_comp_vals.doc_count, query[term].term_freq, len(list_term_occur))
                            soma += wij * wiq
                            print(term, doc_id, soma)

                    documents_weight[doc_id] = soma/doc_value
                    

            # for doc_id, value in self.idx_pre_comp_vals.document_norm.items():
            #     soma = 0
            #     for term, list_term_occur in docs_occur_per_term.items():
            #         wij = 0
            #         wiq = 0
            #         for term_occur in list_term_occur:
            #             if term_occur.doc_id == doc_id:
            #                 wij = self.tf_idf(self.idx_pre_comp_vals.doc_count, term_occur.term_freq, len(list_term_occur))
            #                 wiq = self.tf_idf(self.idx_pre_comp_vals.doc_count, query[term].term_freq, len(list_term_occur))
            #                 soma += wij * wiq
            #                 print(term, doc_id, soma)


            #     if soma != 0:    
            #         documents_weight[doc_id] = soma/value

            #retona a lista de doc ids ordenados de acordo com o TF IDF
            return self.rank_document_ids(documents_weight),documents_weight

