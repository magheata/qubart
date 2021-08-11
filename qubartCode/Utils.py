from datasets import load_dataset


class Utils:
    @staticmethod
    def load_corpus(data_files, ext_type='csv'):
        '''
        '''
        return load_dataset(ext_type, data_files=data_files)

    @staticmethod
    def filter_corpus(corpus, element, filter_with):
        '''
        '''
        return corpus.filter(lambda filtered_corpus: filtered_corpus[element].startswith(filter_with))

    def get_script_episode(self, corpus, season, episode):
        '''
        '''
        data_season = self.filter_corpus(corpus, 'Season', season)
        return self.filter_corpus(data_season, 'Episode', episode)

    @staticmethod
    def get_full_text(input_text):
        return ''.join(input_text)