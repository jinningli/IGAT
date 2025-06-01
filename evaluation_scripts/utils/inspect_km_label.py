import pandas as pd
import os

def remap_km_label(csv_file_path, label_map):
    """
    Reads a CSV file containing a column named 'pred_label'. For each row:
      - Looks up the 'pred_label' in the provided dictionary label_map.
      - Splits that mapped string on the first hyphen into two parts: 
        pred_topic and pred_belief.
      - Adds these two new columns to the DataFrame: pred_topic, pred_belief.
    Saves the modified CSV back to the same file (or you can choose a new filename).
    """
    
    # 1. Load the CSV
    df = pd.read_csv(csv_file_path)
    
    # Make sure these columns exist (in case you re-run the function)
    df["pred_topic"] = ""
    df["pred_belief"] = ""
    
    # 2. For each row, look up the row's pred_label in the label_map, then split on the first hyphen
    for idx in df.index:
        pred_label = df.at[idx, "pred_label"]
        
        if pred_label in label_map:
            # 3. Extract topic and belief from the mapped string
            topic_belief_str = label_map[pred_label]
            
            # We split on the *first* hyphen only
            # If your label_map values are guaranteed to have exactly one hyphen, you can just do `split('-')`
            # If there's a chance of multiple, do `split('-', 1)`
            topic, belief = topic_belief_str.split('--', 1)
            
            df.at[idx, "pred_topic"] = topic.strip()
            df.at[idx, "pred_belief"] = belief.strip()
        else:
            # If pred_label not in label_map, you might want to leave it blank or mark it somehow
            df.at[idx, "pred_topic"] = ""
            df.at[idx, "pred_belief"] = ""
            print(f"pred_label '{pred_label}' not found in label_map")
    
    # 4. Write the updated DataFrame back to CSV
    df.to_csv(csv_file_path, index=False)



def sample_by_label(csv, n=20):
    df = pd.read_csv(csv)
    
    # Only keep ground truth
    df = df[df['is_gt'] == 1]
    print(f"Unique pred_labels in {csv}: {df['pred_label'].unique()}")
    
    # Drop duplicates by 'index_text'
    df = df.drop_duplicates(subset='index_text', keep='first')
    
    # Group by 'pred_label' and sample up to n rows from each
    sampled_df = (
        df.groupby('pred_label', group_keys=False)
          .apply(lambda x: x.sample(n=min(len(x), n), random_state=42)) 
          .reset_index(drop=True)
    )
    
    # Keep only relevant columns
    sampled_df = sampled_df[['index_text', 'text', 'pred_label']]
    
    # Save to new CSV
    sampled_csv = csv.replace('.csv', '_sample20.csv')

    print(f"Afer processing: Unique pred_labels: {sampled_df['pred_label'].nunique()}\n\n\n")
    sampled_df.to_csv(sampled_csv, index=False)


def sample_by_label(csv, n=20):
    directory = 'km_data'
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv') and not file.endswith('_sample20.csv'):
                file_path = os.path.join(root, file)
                # print(file_path)
                sample_by_label(file_path, n=20)


def main():
    roberta_km_phillipines_label_map = {
        # original_label : topic-belief
        "opposing-Crime" : "Government accountability for human rights abuses--End impunity, seek justice",
        "opposing-EDCA": "Philippine sovereignty vs. foreign intervention--Reject foreign intrusion",
        "opposing-Energy_Issues_China": "Global power struggles over sovereignty--Resist oppressive powers",
        "opposing-Social_and_Economic_Issues_Philippines": "Resistance to historical revisionism about the Marcos dictatorship--Reject historical distortion",
        "opposing-United_States_Military_Philippine": "Growing Sino–US tensions over security and sovereignty--Resist infiltration, uphold sovereignty",
        "supportive-Energy_Issues_China": "Debates on China’s governance and sovereignty--Uphold sovereign self-determination",
        "supportive-Social_and_Economic_Issues_Philippines": "Government development under Marcos Jr. and reflection on EDSA People Power--Remember democracy, ensure progress",
        "supportive-United_States_Military_Philippine": "US–Philippines military exercises amid South China Sea tensions--Defend sovereignty from aggression"
    }


    roberta_km_ukraine_war_label_map = {
        "opposing-Ideaology_and_Propaganda": "Criticism of Western narratives in global conflicts--Reject Western hypocrisy",
        "opposing-Western_Involvment_And_Aid": "Criticism of US proxy war in Ukraine--Stop fueling proxy wars",
        "opposing-Western_Sanctions": "Criticism of Western sanctions on Russia harming the Global South--Stop punishing Global South",
        "supportive-Ideaology_and_Propaganda": "Competing narratives of Nazism and Russophobia in the Russia–Ukraine war--Reject hateful propaganda",
        "supportive-War_Crime": "War crimes and forced conscription in Russia’s invasion of Ukraine--End atrocities, seek justice",
        "supportive-War_Responsibility":"NATO and Europe’s role in Russia’s invasion of Ukraine--Stand firm against aggression",
        "supportive-Western_Involvment_And_Aid": "Arming Ukraine to repel Russian invasion--Arm Ukraine, stop aggression"
    }

    roberta_km_us_election_label_map = {
        "opposing-Candidate_Advocacy": "Criticism of Trump and GOP complicity, calls for accountability--Hold Trump accountable now",
        "opposing-Election_Legitimacy": "Post-election litigation to overturn 2020 results--Respect lawful election results",
        "supportive-Candidate_Advocacy": "Criticism of Trump’s refusal to concede and calls to uphold democratic norms--Honor election results now",
        "supportive-Election_Legitimacy": "Efforts to overturn the 2020 presidential election and calls to respect certified results--Respect legitimate election outcomes"
    }


    tweet_roberta_km_phillipines_label_map = {
        "opposing-Crime": "State-perpetrated violence and calls for accountability--End extrajudicial killings now",
        "opposing-EDCA": "Debate over US–Philippines EDCA agreement--Reject EDCA, preserve sovereignty",
        "opposing-Insurgent_Threats": "Critiques of authoritarian tactics and foreign meddling--Resist authoritarian foreign interference",
        "opposing-Labor_and_Migration_China": "Concerns over China’s global influence and expansionism--Resist authoritarian foreign expansion",
        "opposing-Social_and_Economic_Issues_Philippines": "Criticism of Marcos’ economic claims and calls for transparency--Demand honesty, reject falsehoods",
        "opposing-United_States_Military_Philippine": "Philippine sovereignty amid US–China tensions--Defend sovereignty, reject foreign conflict",
        "supportive-Crime": "Investigations into violent incidents and humanitarian pleas--Stop violence, seek justice",
        "supportive-Energy_Issues_China": "China’s growing global role and tensions in the South China Sea--Embrace cooperation, prevent conflict",
        "supportive-Social_and_Economic_Issues_Philippines": "Marcos administration’s development initiatives and public concerns--Pursue inclusive growth responsibly",
        "supportive-United_States_Military_Philippine": "Military buildup in the Philippines amid US–China tensions--Reject escalation, pursue peace"
    }

    tweet_roberta_km_ukraine_war_label_map = {
        "opposing-Ideaology_and_Propaganda": "Alleged Western hypocrisy over Nazi elements and Russophobia in Ukraine--Stop enabling Nazi proxies",
        "opposing-Western_Involvment_And_Aid": "Criticism of U.S. proxy war in Ukraine--Stop fueling proxy war",
        "opposing-Western_Sanctions": "Western sanctions on Russia and their global fallout--Stop punishing Global South",
        "supportive-Ideaology_and_Propaganda": "Accusations of Western double standards on global crises (Ukraine vs. Tigray) and Russophobia--End selective global concern",
        "supportive-War_Crime": "War crimes and atrocities in Russia’s invasion of Ukraine--End aggression, demand accountability",
        "supportive-War_Responsibility": "Condemnation of Russia’s invasion of Ukraine and calls for global support--End war, support Ukraine",
        "supportive-Western_Involvment_And_Aid": "Calls to supply Ukraine with advanced weapons to repel Russia--Arm Ukraine, end aggression"
    }

    tweet_roberta_km_us_election_label_map = {
        "opposing-Candidate_Advocacy": " Condemnation of Trump’s final days and support for Biden’s presidency--Remove Trump, trust Biden",
        "opposing-Election_Legitimacy": "Empowering progressive leaders and scientific expertise--Champion women, trust science",
        "supportive-Candidate_Advocacy": "Condemnation of Trump’s subversion of democracy and calls for accountability--Defend democracy, demand accountability",
        "supportive-Election_Legitimacy": "Attempts to overturn 2020 election and calls to uphold results--Respect lawful election outcome"
    }

    twhin_bert_phillipines_map = {
        "opposing-Crime": "Activism against killings and rights violations--Stop killings, demand justice",
        "opposing-EDCA": "Debate over expanded EDCA and Philippine sovereignty--Reject foreign military dominance",
        "opposing-Energy_Issues_China": "Growing US–China tensions and militarization--Resist militarization, seek diplomacy",
        "opposing-Labor_and_Migration_China": "Debates on China’s governance, territorial claims, and global influence--Reject hypocrisy, demand fairness",
        "opposing-Social_and_Economic_Issues_Philippines": "Criticism of Marcos regime’s corruption and disinformation--Reject dictatorship, seek accountability",
        "opposing-United_States_Military_Philippine": "US–Philippines alignment versus China’s regional ambitions--Balance interests, prevent conflict",
        "supportive-Social_and_Economic_Issues_Philippines": "Marcos administration’s economic policies vs. EDSA legacy--Ensure transparency, respect democracy",
        "supportive-United_States_Military_Philippine": "Heightened US–Philippines military cooperation amid South China Sea tensions--Strengthen defense, maintain peace"
        
    }

    twhin_bert_ukraine_war_map = {
        "opposing-Ideaology_and_Propaganda": "Claims of Nazi influence in Ukraine and Western hypocrisy--Reject Nazism, oppose hypocrisy",
        "opposing-Western_Involvment_And_Aid": "Criticism of the US-led proxy war in Ukraine--Stop fueling proxy wars",
        "opposing-Western_Sanctions": "Western sanctions on Russia and their global repercussions--Stop punishing Global South",
        "supportive-Ideaology_and_Propaganda": "Condemnation of Russia’s invasion and alleged Russophobia claims--End war, defend Ukraine",
        "supportive-War_Crime": "Accusations of Russian war crimes in Ukraine--Stop atrocities, seek justice",
        "supportive-War_Responsibility": "Condemnation of Russia’s invasion, support for Ukraine’s defense--Resist aggression, aid Ukraine",
        "supportive-Western_Involvment_And_Aid": "Calls to strengthen Ukraine’s military against Russia--Arm Ukraine, halt aggression"
    }

    twhin_bert_us_election_map = {
        "opposing-Candidate_Advocacy": "Criticisms of GOP hypocrisy and support for Biden's leadership--Reject hypocrisy, trust Biden",
        "opposing-Election_Legitimacy": "Attempts to overturn the 2020 presidential election and the courts’ repeated rejections--Uphold fair election results",
        "supportive-Candidate_Advocacy": "COVID-19 responses under the Biden administration--Trust science, save lives",
        "supportive-Election_Legitimacy": "Certification of Biden’s 2020 presidential victory--Accept legitimate election results"
    }

    remap_km_label('km_data/twhin-bert-km/us-elections_ft10_labeled_sample20.csv', twhin_bert_us_election_map)


if __name__ == '__main__':
    main()
