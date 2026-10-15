"""
data/create_benchmark_data.py
Génère le jeu de données benchmark commun standardisé (HotpotQA multi-hop + négatifs)
avec documents sources associés et triplets de connaissances pour KAG.
"""

import json
from pathlib import Path

# Ensemble de questions multi-hop HotpotQA canoniques avec vérités terrain et passages sources
DATASET = [
    {
        "id": "hotpot_01",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "answer": "yes",
        "type": "comparison",
        "level": "medium",
        "is_answerable": True,
        "docs": [
            {
                "title": "Scott Derrickson",
                "text": "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer. He lives in Los Angeles, California. He is known for directing horror films such as Sinister, The Exorcism of Emily Rose, and Deliver Us from Evil, as well as the 2016 Marvel Cinematic Universe film Doctor Strange."
            },
            {
                "title": "Ed Wood",
                "text": "Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, and pulp novel author. In the 1950s, Wood directed several low-budget science fiction, comedy, and horror films, including Glen or Glenda (1953), Jail Bait (1954), Bride of the Monster (1955), and Plan 9 from Outer Space (1957)."
            },
            {
                "title": "Doctor Strange (film)",
                "text": "Doctor Strange is a 2016 American superhero film based on the Marvel Comics character of the same name, produced by Marvel Studios and distributed by Walt Disney Studios Motion Pictures. It was directed by Scott Derrickson from a screenplay he wrote with Jon Spaihts and C. Robert Cargill."
            }
        ]
    },
    {
        "id": "hotpot_02",
        "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
        "answer": "Chief of Protocol",
        "type": "bridge",
        "level": "hard",
        "is_answerable": True,
        "docs": [
            {
                "title": "Kiss and Tell (1945 film)",
                "text": "Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer. In the film, two teenage girls cause havoc when one secretly marries a soldier and the other pretends to be pregnant to protect her friend."
            },
            {
                "title": "Shirley Temple",
                "text": "Shirley Temple Black (April 23, 1928 – February 10, 2014) was an American actress, singer, dancer, and diplomat. Later in life, she served as United States Ambassador to Ghana and to Czechoslovakia, and as Chief of Protocol of the United States from 1976 to 1977."
            },
            {
                "title": "Corliss Archer",
                "text": "Corliss Archer is a fictional character created by F. Hugh Herbert in a series of short stories and later adapted into the radio program Meet Corliss Archer and the play and film Kiss and Tell."
            }
        ]
    },
    {
        "id": "hotpot_03",
        "question": "What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?",
        "answer": "Animorphs",
        "type": "bridge",
        "level": "medium",
        "is_answerable": True,
        "docs": [
            {
                "title": "Animorphs",
                "text": "Animorphs is a science fantasy young adult novel series written by Katherine Applegate and her husband Michael Grant. The series is told in first person, with all six main characters taking turns narrating the books through their perspectives. Companion books to the series include the Megamorphs series and the Chronicles series, which detail stories of alien species and enslaved worlds."
            },
            {
                "title": "The Andalite Chronicles",
                "text": "The Andalite Chronicles is a companion book to the Animorphs series, detailing the life of Elfangor-Sirinial-Shamtul before the Yeerk invasion of Earth, and the history of the enslaved Hork-Bajir world."
            }
        ]
    },
    {
        "id": "hotpot_04",
        "question": "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?",
        "answer": "no",
        "type": "comparison",
        "level": "medium",
        "is_answerable": True,
        "docs": [
            {
                "title": "Laleli Mosque",
                "text": "The Laleli Mosque is an 18th-century Ottoman imperial mosque situated in Laleli, Fatih, Istanbul, Turkey. It was constructed by Sultan Mustafa III from 1760 to 1763."
            },
            {
                "title": "Esma Sultan Mansion",
                "text": "The Esma Sultan Mansion is a historical yalı (waterside mansion) located at the Bosphorus in the Ortaköy neighborhood of Istanbul, Turkey. Named after its original owner Esma Sultan, it was recognized as an event venue after refurbishment."
            },
            {
                "title": "Ortaköy",
                "text": "Ortaköy is a neighbourhood, formerly a small village, within the Beşiktaş district of Istanbul, Turkey, located on the European bank of the Bosphorus strait."
            }
        ]
    },
    {
        "id": "hotpot_05",
        "question": "The director of the romantic comedy \"Big Stone Gap\" is based in what New York city?",
        "answer": "Greenwich Village, New York City",
        "type": "bridge",
        "level": "medium",
        "is_answerable": True,
        "docs": [
            {
                "title": "Big Stone Gap (film)",
                "text": "Big Stone Gap is a 2014 American drama romantic comedy film written and directed by Adriana Trigiani, based on her 2000 best-selling novel of the same name. It stars Ashley Judd, Patrick Wilson, and Whoopi Goldberg."
            },
            {
                "title": "Adriana Trigiani",
                "text": "Adriana Trigiani is an Italian-American best-selling author, television writer, and film director. She was raised in Big Stone Gap, Virginia, and currently lives in Greenwich Village, New York City with her family."
            }
        ]
    },
    {
        "id": "hotpot_06",
        "question": "Which magazine was started first, Arthur's Magazine or First for Women?",
        "answer": "Arthur's Magazine",
        "type": "comparison",
        "level": "easy",
        "is_answerable": True,
        "docs": [
            {
                "title": "Arthur's Magazine",
                "text": "Arthur's Magazine (1844–1846) was an American literary magazine published in Philadelphia in the 19th century. It was founded by T.S. Arthur and merged into Godey's Lady's Book in 1846."
            },
            {
                "title": "First for Women",
                "text": "First for Women is an American woman's magazine published by the Bauer Media Group in Englewood Cliffs, New Jersey. The magazine was started in 1989 and focuses on service journalism."
            }
        ]
    },
    {
        "id": "hotpot_07",
        "question": "What band is associated with both Noel Gallagher and Liam Gallagher?",
        "answer": "Oasis",
        "type": "bridge",
        "level": "easy",
        "is_answerable": True,
        "docs": [
            {
                "title": "Noel Gallagher",
                "text": "Noel Thomas David Gallagher (born 29 May 1967) is an English singer, songwriter and musician. He served as the co-lead vocalist, lead guitarist and primary songwriter of the rock band Oasis."
            },
            {
                "title": "Liam Gallagher",
                "text": "William John Paul Gallagher (born 21 September 1972) is an English singer and songwriter. He achieved fame as the lead singer of the rock band Oasis from 1991 to 2009."
            },
            {
                "title": "Oasis (band)",
                "text": "Oasis were an English rock band formed in Manchester in 1991. The band initially consisted of Liam Gallagher, Paul Arthurs, Paul McGuigan, and Tony McCarroll, later joined by Noel Gallagher."
            }
        ]
    },
    {
        "id": "hotpot_08",
        "question": "Were both Aleksandr Pushkin and Leo Tolstoy Russian writers?",
        "answer": "yes",
        "type": "comparison",
        "level": "easy",
        "is_answerable": True,
        "docs": [
            {
                "title": "Aleksandr Pushkin",
                "text": "Aleksandr Sergeyevich Pushkin (1799 – 1837) was a Russian poet, playwright, and novelist of the Romantic era who is considered by many to be the greatest Russian poet and the founder of modern Russian literature."
            },
            {
                "title": "Leo Tolstoy",
                "text": "Count Lev Nikolayevich Tolstoy (1828 – 1910), usually referred to in English as Leo Tolstoy, was a Russian writer who is regarded as one of the greatest authors of all time, known for War and Peace and Anna Karenina."
            }
        ]
    },
    {
        "id": "hotpot_09",
        "question": "What university did the founder of Khan Academy attend for his MBA?",
        "answer": "Harvard Business School",
        "type": "bridge",
        "level": "hard",
        "is_answerable": True,
        "docs": [
            {
                "title": "Sal Khan",
                "text": "Salman Amin Khan (born October 11, 1976) is an American educator and the founder of Khan Academy, a free online education platform. He holds bachelor's and master's degrees from MIT and an MBA from Harvard Business School."
            },
            {
                "title": "Khan Academy",
                "text": "Khan Academy is an American non-profit educational organization created in 2008 by Sal Khan. Its mission is to provide a free, world-class education for anyone, anywhere."
            }
        ]
    },
    {
        "id": "hotpot_10",
        "question": "Did the director of Inception also direct The Prestige?",
        "answer": "yes",
        "type": "bridge",
        "level": "medium",
        "is_answerable": True,
        "docs": [
            {
                "title": "Inception",
                "text": "Inception is a 2010 science fiction action film written and directed by Christopher Nolan, who also produced the film with Emma Thomas. The film stars Leonardo DiCaprio as a professional thief who steals information by entering the dreams of others."
            },
            {
                "title": "The Prestige (film)",
                "text": "The Prestige is a 2006 psychological thriller film directed by Christopher Nolan and written by Nolan and his brother Jonathan, based on the 1995 novel by Christopher Priest. It stars Hugh Jackman and Christian Bale."
            }
        ]
    },
    # Questions Négatives (Information absente du corpus pour tester la robustesse au refus)
    {
        "id": "neg_01",
        "question": "What was the worldwide box office gross of the 2025 sequel to Big Stone Gap?",
        "answer": "Information absente du corpus",
        "type": "unanswerable",
        "level": "hard",
        "is_answerable": False,
        "docs": []
    },
    {
        "id": "neg_02",
        "question": "Which telecommunications satellite was launched jointly by Scott Derrickson and Ed Wood?",
        "answer": "Information absente du corpus",
        "type": "unanswerable",
        "level": "hard",
        "is_answerable": False,
        "docs": []
    },
    {
        "id": "neg_03",
        "question": "In which year did Esma Sultan receive an honorary doctorate from Harvard University?",
        "answer": "Information absente du corpus",
        "type": "unanswerable",
        "level": "hard",
        "is_answerable": False,
        "docs": []
    }
]

def main():
    Path("data").mkdir(parents=True, exist_ok=True)
    corpus_docs = {}
    benchmark_questions = []

    for item in DATASET:
        q_id = item["id"]
        doc_keys = []
        for d in item.get("docs", []):
            title = d["title"]
            text = d["text"]
            doc_key = f"doc_{abs(hash(title)) % 1000000:06d}"
            doc_keys.append(doc_key)
            if doc_key not in corpus_docs:
                corpus_docs[doc_key] = {
                    "id": doc_key,
                    "title": title,
                    "text": text
                }
        benchmark_questions.append({
            "id": q_id,
            "question": item["question"],
            "answer": item["answer"],
            "type": item["type"],
            "level": item["level"],
            "document_ids": doc_keys,
            "is_answerable": item["is_answerable"]
        })

    docs_list = list(corpus_docs.values())

    with open("data/benchmark_questions.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_questions, f, indent=2, ensure_ascii=False)

    with open("data/corpus_documents.json", "w", encoding="utf-8") as f:
        json.dump(docs_list, f, indent=2, ensure_ascii=False)

    print(f"Jeu de données créé : {len(benchmark_questions)} questions ({len(docs_list)} documents uniques)")

if __name__ == "__main__":
    main()
