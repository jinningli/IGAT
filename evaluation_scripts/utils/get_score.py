import os


def get_scores(dir):
    scores = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file == "score.txt":
                subdir = os.path.basename(root)
                score_path = os.path.join(root, file)
                with open(score_path, 'r') as f:
                    score_content = f.read()
                formatted_score = f"{subdir}:\n\n{score_content}"
                scores.append(formatted_score)

    # Write all scores to a single file
    with open("all_scores.txt", 'w') as f:
        f.write('\n\n'.join(scores))

    return scores



if __name__ == "__main__":
    get_scores("/home/path1/incas/src/belief_embedding/infovgae/demo_result")