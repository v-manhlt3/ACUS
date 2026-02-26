import os
# from clair_a import clair_a
import json
import tqdm
from mace.mace_metric.mace import mace

# os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'
def evaluate_pred(json_file):
    list_score = []
    with open(json_file, "r") as f:
        data = json.load(f)
        keys = data.keys()
        for key in tqdm.tqdm(list(keys)[:100]):
            audio_fp = os.path.join("AudioCaps", "test", key)
            cand = data[key]["prediction"]
            ref = data[key]["references"]
            # score = clair_a(cand, ref, model='openai/gpt-4o-2024-08-06')
            # list_score.append(score[0]
            score = mace(method='combined', candidates=[cand], mult_references=[ref], audio_paths=[audio_fp])

            print(score[0]['mace'].detach().cpu().item())
            list_score.append(score[0]['mace'].detach().cpu().item())
    return sum(list_score)/len(list_score)
            # print(data[key]["prediction"])


if __name__ =="__main__":
    # candidates = 'Rain is splashing on a surface while rustling occurs and a car door shuts, and traffic is discernible in the distance'
    # references = ['Rain falls soft and steadily and a person closes a car door and walks away through leaves',
    #             'Rain falling followed by fabric rustling and footsteps shuffling then a vehicle door opening and closing as plastic crinkles',
    #             'Rain falling followed by footsteps walking on grass then a vehicle door opening then closing',
    #             'Light rainfall together with rustling']

    # score = clair_a(candidates, references, model='openai/gpt-4o-2024-08-06')
    # print(score)
    avg_score = evaluate_pred("pred-fold/[Eval]enclap-AudioCaps-SW.json")
    print("CLAIR-A score: ", avg_score)