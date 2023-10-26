from torch import nn
from torch.utils import data
from naive_gpt import loaders


# not used
subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

# not used
categories = {
    "Humanities": ["history", "philosophy", "law"],
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "Social Sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "Others": ["other", "business", "health"],
}


class MMLUPrompt(nn.Module):
    prompt = "The following are multiple choice questions (with answers) about"

    def forward(self, item: tuple):
        row, path = item
        if len(row) != 6:
            raise RuntimeError

        # subject
        filename = path.split('/')[-1]
        subject = ' '.join(
            filename.split('_')[:-1]
        )

        # question
        choices = ['A', 'B', 'C', 'D']
        question = '{}\n'.format(row[0])
        question += '\n'.join([
            '{}. {}'.format(choices[i], row[1 + i])
            for i in range(4)
        ])

        # text
        text = '{} {}\n{}\nAnswer: {}'.format(
            self.prompt, subject, question, row[-1]
        )
        return text


class MMLUDataset(data.IterableDataset):
    def __init__(self,
                 root: str,
                 mode: str,
                 n_shots: int = 0,
                 shuffle: bool = True,
                 min_length: int = 64,
                 buffer_size: int = 16384,
                 return_path: bool = False,
                 text_transform: callable = None,
                 path_transform: callable = None):
        data.IterableDataset.__init__(self)
        #
        self.n_shots = n_shots
        self.return_path = return_path
        self.text_transform = text_transform
        self.path_transform = path_transform
        #
        datapath = {
            'test': 'test',
            'valid': 'val',
            'train': 'auxiliary_train'
        }
        if mode not in datapath:
            raise RuntimeError
        #
        self.context = loaders.TextFolder(
            root='{}/mmlu/dev'.format(root.rstrip('/')),
            reader='csv', shuffle=True, skip_lines=0,
            min_length=min_length, buffer_size=buffer_size,
            return_path=False, append_path=True,
            text_transform=MMLUPrompt()
        )
        self.dataset = loaders.TextFolder(
            root='{}/mmlu/{}'.format(
                root.rstrip('/'), datapath[mode]
            ),
            reader='csv', shuffle=shuffle, skip_lines=0,
            min_length=min_length, buffer_size=buffer_size,
            return_path=return_path, append_path=True,
            text_transform=MMLUPrompt()
        )

    def __iter__(self):
        ctx_iter = iter(self.context)
        for item in self.dataset:
            #
            if self.return_path:
                content, filename = item
            else:
                content, filename = item, None
            # few-shot
            prompt = []
            for _ in range(self.n_shots):
                prompt.append(next(ctx_iter))
            prompt.append(content)
            content = '\n\n'.join(prompt)
            # transform after few-shot
            if self.text_transform:
                content = self.text_transform(content)
            if self.path_transform:
                filename = self.path_transform(filename)
            # return final result
            if self.return_path:
                yield content, filename
            else:
                yield content
