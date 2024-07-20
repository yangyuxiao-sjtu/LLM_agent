from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import sys
from lgp.env.alfred.segmentation_definitions import _INTERACTIVE_OBJECTS
import json


class LLM_HLP_Generator:
    def __init__(
        self,
        knn_data_path="/mnt/sda/yuxiao_code/LLM_subgoal/prompts/llm_samples.json",
        emb_model_name="paraphrase-MiniLM-L6-v2",
    ):
        self.sentence_embedder = SentenceTransformer(emb_model_name)
        with open(knn_data_path, "r", encoding="utf-8") as f:
            self.knn_set = json.load(f)

    def knn_retrieval(self, curr_task, k):
        # Find K train examples with closest sentence embeddings to test example

        traj_emb = self.sentence_embedder.encode(curr_task)
        topK = []
        for trainItem in self.knn_set:

            train_emb = self.sentence_embedder.encode(trainItem[0]["task"])

            dist = -1 * cos_sim(traj_emb, train_emb)
            topK.append((trainItem, dist))

        topK = sorted(topK, key=lambda x: x[1])
        topK = topK[:k]
        return [entry[0] for entry in topK]


if __name__ == "__main__":
    print(_INTERACTIVE_OBJECTS)
    # knn = LLM_HLP_Generator()
    # task = 'get the laptop, turn the lamp on'
    # ans = knn.knn_retrieval(task,5)
    # for item in ans:
    #     print(item[0]['task'])
