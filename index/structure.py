from ast import Break
import sys
from IPython.display import clear_output
from typing import List, Set, Union
from abc import abstractmethod
from functools import total_ordering
from os import path
import os
import pickle
import gc


class Index:
    def __init__(self):
        self.dic_index = {}
        self.set_documents = set()

    def index(self, term: str, doc_id: int, term_freq: int):
        if term not in self.dic_index:
            int_term_id = len(self.dic_index)
            self.dic_index[term] = self.create_index_entry(int_term_id)
        else:
            int_term_id = self.get_term_id(term)

        self.add_index_occur(self.dic_index[term], doc_id, int_term_id, term_freq)
        self.set_documents.add(doc_id)

    @property
    def vocabulary(self) -> List[str]:
        return self.dic_index.keys()

    @property
    def document_count(self) -> int:
        return len(self.set_documents)

    @abstractmethod
    def get_term_id(self, term: str):
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def create_index_entry(self, termo_id: int):
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def add_index_occur(self, entry_dic_index, doc_id: int, term_id: int, freq_termo: int):
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def get_occurrence_list(self, term: str) -> List:
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def document_count_with_term(self, term: str) -> int:
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    def finish_indexing(self):
        pass

    def write(self, arq_index: str):
        with open(arq_index, 'wb') as idx_file:
            pickle.dump(self, idx_file)


    @staticmethod
    def read(arq_index: str):
        with open(arq_index, 'rb') as idx_file:
            sys.path.append('index')
            return pickle.load(idx_file)

    def __str__(self):
        arr_index = []
        for str_term in self.vocabulary:
            arr_index.append(f"{str_term} -> {self.get_occurrence_list(str_term)}")

        return "\n".join(arr_index)

    def __repr__(self):
        return str(self)


@total_ordering
class TermOccurrence:
    def __init__(self, doc_id: int, term_id: int, term_freq: int):
        self.doc_id = doc_id
        self.term_id = term_id
        self.term_freq = term_freq

    def write(self, idx_file):
       idx_file.write(self.doc_id.to_bytes(4,byteorder="big"))
       idx_file.write(self.term_id.to_bytes(4,byteorder="big"))
       idx_file.write(self.term_freq.to_bytes(4,byteorder="big"))

    def __hash__(self):
        return hash((self.doc_id, self.term_id))

    def __eq__(self, other_occurrence: "TermOccurrence"):
        if other_occurrence is None:
            return False
        return self.term_id == other_occurrence.term_id and self.doc_id == other_occurrence.doc_id

    def __lt__(self, other_occurrence: "TermOccurrence"):
        if other_occurrence is None:
            return True
        return self.term_id < other_occurrence.term_id if self.term_id != other_occurrence.term_id else self.doc_id < other_occurrence.doc_id

    def __str__(self):
        return f"( doc: {self.doc_id} term_id:{self.term_id} freq: {self.term_freq})"

    def __repr__(self):
        return str(self)


# HashIndex é subclasse de Index
class HashIndex(Index):
    def get_term_id(self, term: str):
        return self.dic_index[term][0].term_id

    def create_index_entry(self, termo_id: int) -> List:
        return []

    def add_index_occur(self, entry_dic_index: List[TermOccurrence], doc_id: int, term_id: int, term_freq: int):
        entry_dic_index.append(TermOccurrence(doc_id, term_id, term_freq))

    def get_occurrence_list(self, term: str) -> List:
        return self.dic_index[term] if term in self.dic_index else []

    def document_count_with_term(self, term: str) -> int:
        return len(self.get_occurrence_list(term))


class TermFilePosition:
    def __init__(self, term_id: int, term_file_start_pos: int = None, doc_count_with_term: int = None):
        self.term_id = term_id

        # a serem definidos após a indexação
        self.term_file_start_pos = term_file_start_pos
        self.doc_count_with_term = doc_count_with_term

    def __str__(self):
        return f"term_id: {self.term_id}, doc_count_with_term: {self.doc_count_with_term}, term_file_start_pos: {self.term_file_start_pos}"

    def __repr__(self):
        return str(self)


class FileIndex(Index):
    TMP_OCCURRENCES_LIMIT = 1000000

    def __init__(self):
        super().__init__()

        self.lst_occurrences_tmp = [None]*FileIndex.TMP_OCCURRENCES_LIMIT
        self.idx_file_counter = 0
        # self.str_idx_file_name = "occur_idx_file"
        self.str_idx_file_name = None

        # metodos auxiliares para verifica o tamanho da lst_occurrences_tmp
        self.idx_tmp_occur_last_element  = -1
        self.idx_tmp_occur_first_element = 0

    def get_term_id(self, term: str):
        return self.dic_index[term].term_id

    def create_index_entry(self, term_id: int) -> TermFilePosition:
        return TermFilePosition(term_id)

    def add_index_occur(self, entry_dic_index: TermFilePosition, doc_id: int, term_id: int, term_freq: int):
        #complete aqui adicionando um novo TermOccurrence na lista lst_occurrences_tmp
        #não esqueça de atualizar a(s) variável(is) auxiliares apropriadamente

        self.lst_occurrences_tmp[self.idx_tmp_occur_last_element + 1] = TermOccurrence(doc_id, term_id, term_freq)
        self.idx_tmp_occur_last_element += 1

        if self.get_tmp_occur_size() > FileIndex.TMP_OCCURRENCES_LIMIT:
            self.save_tmp_occurrences()

    def next_from_list(self) -> TermOccurrence:
        # obtenha o proximo da lista e armazene em nex_occur
        # não esqueça de atualizar a(s) variável(is) auxiliares apropriadamente
        if self.get_tmp_occur_size() <= 0:
            return None

        next_occur = self.lst_occurrences_tmp[self.idx_tmp_occur_first_element]
        self.lst_occurrences_tmp[self.idx_tmp_occur_first_element] = None
        self.idx_tmp_occur_first_element += 1

        return next_occur

    def get_tmp_occur_size(self):
        return (self.idx_tmp_occur_last_element - self.idx_tmp_occur_first_element) + 1



    def next_from_file(self, file_pointer) -> TermOccurrence:
        if file_pointer is None:
            return None

        bytes_doc_id = file_pointer.read(4)
        bytes_term_id = file_pointer.read(4)
        bytes_term_freq = file_pointer.read(4)


        if not bytes_doc_id or not bytes_term_id or not bytes_term_freq:
            return None

        doc_id = int.from_bytes(bytes_doc_id,byteorder='big')
        term_id = int.from_bytes(bytes_term_id,byteorder='big')
        term_freq = int.from_bytes(bytes_term_freq,byteorder='big')

        return TermOccurrence(doc_id, term_id, term_freq)

    def save_tmp_occurrences(self):

        # Ordena pelo term_id, doc_id
        #    Para eficiência, todo o código deve ser feito com o garbage collector desabilitado gc.disable()
        gc.disable()

        # Aparentemente o python nao connsegue ordenar um intervalo de um array, entao ele precisou ser extraido
        valid_values = self.lst_occurrences_tmp[self.idx_tmp_occur_first_element:self.
                idx_tmp_occur_last_element+1]
        valid_values.sort()
        self.lst_occurrences_tmp[self.idx_tmp_occur_first_element:self.
                idx_tmp_occur_last_element+1] = valid_values

        current_file = None
        if self.str_idx_file_name is None:
            self.str_idx_file_name= f'occur_index_{self.idx_file_counter}'
        else:
            current_file = open(self.str_idx_file_name, 'rb')
            self.idx_file_counter += 1
            self.str_idx_file_name = f'occur_index_{self.idx_file_counter}'

        new_file = open(f'occur_index_{self.idx_file_counter}', 'wb')

        file_idx = self.next_from_file(current_file)
        list_idx = self.next_from_list()
        while True:
            if file_idx is None and list_idx is None:
                break

            lower_idx = None
            if list_idx < file_idx or file_idx is None:
                lower_idx = list_idx
                list_idx = self.next_from_list()
            else:
                lower_idx = file_idx
                file_idx = self.next_from_file(current_file)
            lower_idx.write(new_file)

        self.lst_occurrences_tmp = [None] * FileIndex.TMP_OCCURRENCES_LIMIT
        self.idx_tmp_occur_last_element  = -1
        self.idx_tmp_occur_first_element = 0

        new_file.close()
        if current_file is not None:
            current_file.close()
        """comparar sempre a primeira posição
        da lista com a primeira posição do arquivo usando os métodos next_from_list e next_from_file
        e use o método write do TermOccurrence para armazenar cada ocorrencia do novo índice ordenado"""

        gc.enable()

    def finish_indexing(self):
        if len(self.lst_occurrences_tmp) > 0:
            self.save_tmp_occurrences()

        # Sugestão: faça a navegação e obetenha um mapeamento
        # id_termo -> obj_termo armazene-o em dic_ids_por_termo
        # obj_termo é a instancia TermFilePosition correspondente ao id_termo
        dic_ids_por_termo = {}
        for str_term, obj_term in self.dic_index.items():
            dic_ids_por_termo[obj_term.term_id] = obj_term

        with open(self.str_idx_file_name, 'rb') as idx_file:
            # navega nas ocorrencias para atualizar cada termo em dic_ids_por_termo
            # apropriadamente
            current_pos = 0
            ocurrence = self.next_from_file(idx_file)
            while ocurrence is not None:
                term_file_pos = dic_ids_por_termo[ocurrence.term_id]

                if term_file_pos.doc_count_with_term is None:
                    term_file_pos.doc_count_with_term = 0

                if term_file_pos.term_file_start_pos is None:
                    term_file_pos.term_file_start_pos = current_pos

                term_file_pos.doc_count_with_term += 1
                current_pos += 12

                ocurrence = self.next_from_file(idx_file)






    def get_occurrence_list(self, term: str) -> List:
        result = []

        if term not in self.dic_index:
            return result

        term_file_pos = self.dic_index[term]

        with open(self.str_idx_file_name, 'rb') as idx_file:
            idx_file.seek(term_file_pos.term_file_start_pos)

            i = 0
            next = self.next_from_file(idx_file)
            while i < term_file_pos.doc_count_with_term and next.term_id == term_file_pos.term_id:
                result.append(next)
                next = self.next_from_file(idx_file)
                i += 1

        return result

    def document_count_with_term(self, term: str) -> int:
        return len(self.get_occurrence_list(term))
