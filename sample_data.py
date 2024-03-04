sample_data = [
    "It was horrible movie, waste of time.",
    "I left the cinema halfway through the film, it was unbearable.",
    "It was pretty lame, I expected better.",
    "Bad editing and not funny at all.",
    "The acting was atrocious, I cringed throughout.",
    "I regret watching this movie, total disappointment.",
    "The plot was predictable and uninspiring.",
    "I couldn't wait for it to end, absolute boredom.",
    "The characters were poorly developed and uninteresting.",
    "I don't understand the hype, it was mediocre at best.",
    "What a disaster, I can't believe I wasted money on this.",
    "I've seen better acting in high school plays.",
    "This movie was a letdown, don't bother watching it.",
    "I'm baffled by the positive reviews, it's just bad.",
    "It felt like the longest movie of my life, so boring.",
    "I was hoping it would get better, but it never did.",
    "The dialogue was cringeworthy, I felt embarrassed.",
    "I couldn't connect with any of the characters.",
    "The script was terrible, did they even try?",
    "I'd rather watch paint dry than sit through this again.",
    "This movie was a waste of talent.",
    "I can't believe I wasted two hours of my life on this.",
    "It was painful to watch, I'm still recovering.",
    "The plot holes were glaringly obvious.",
    "I've never been so disappointed in a film.",
    "I'm going to pretend I never saw this movie.",
    "I can't think of anything positive to say about this.",
    "This movie was an insult to my intelligence.",
    "I've seen better acting in commercials.",
    "It's rare for a movie to be this bad.",
    "I'm genuinely upset by how bad this was.",
    "I've never walked out of a movie until now.",
    "It's a miracle I made it through the whole thing.",
    "I wouldn't recommend this to my worst enemy.",
    "I expected better, what a letdown.",
    "The direction was all over the place.",
    "I'm embarrassed to admit I watched this.",
    "This movie ruined my day.",
    "I'm angry that I wasted money on this.",
    "I'd rather do anything else than watch this again.",
    "I've never been so disappointed in a film.",
    "I'm going to pretend I never saw this movie.",
    "I can't think of anything positive to say about this.",
    "This movie was an insult to my intelligence.",
    "I've seen better acting in commercials.",
    "It's rare for a movie to be this bad.",
    "I'm genuinely upset by how bad this was.",
    "I've never walked out of a movie until now.",
    "It's a miracle I made it through the whole thing.",
    "I wouldn't recommend this to my worst enemy.",
    "I expected better, what a letdown.",
    "The direction was all over the place.",
    "I'm embarrassed to admit I watched this.",
    "This movie ruined my day.",
    "I'd rather do anything else than watch this again.",
    "I'm angry that I wasted money on this, i could have just sat home and watch paint on wall instead.",
    "Omg the worst i have ever seen",
    "Not good at all",
    "Have never seen anything this bad",
    "I do not recommend this to anyone",
    "Would not even have watched this if i got payed",
    "I love this movie, it's my favorite!",
    "The acting in this film is superb.",
    "What an amazing performance by the cast!",
    "This movie exceeded all my expectations.",
    "I highly recommend this film to everyone.",
    "The cinematography is breathtaking.",
    "The plot twists kept me on the edge of my seat.",
    "I couldn't stop laughing throughout the entire movie.",
    "The soundtrack perfectly complements the story.",
    "A must-watch for any cinema lover.",
    "I absolutely adored this movie!",
    "This film is a masterpiece.",
    "The special effects are out of this world.",
    "I'm blown away by how good this movie is.",
    "It's a feel-good movie that leaves you smiling.",
    "I'm so glad I watched this, it's fantastic.",
    "The characters are well-developed and relatable.",
    "I couldn't tear my eyes away from the screen.",
    "The dialogue is witty and clever.",
    "This movie touched my heart.",
    "I'm speechless, this movie is incredible.",
    "The emotional depth of this film is remarkable.",
    "I was captivated from start to finish.",
    "This film deserves all the praise it's getting.",
    "I'm already planning to watch it again.",
    "This movie made me cry tears of joy.",
    "I'm recommending this to all my friends.",
    "The performances were top-notch.",
    "This movie is a work of art.",
    "I'm still thinking about this movie days later.",
    "I was completely immersed in the story.",
    "The directing is flawless.",
    "This movie has everything: action, drama, romance.",
    "I was pleasantly surprised by how much I enjoyed it.",
    "The humor in this film is spot-on.",
    "This movie is a game-changer.",
    "I felt a range of emotions while watching this.",
    "I can't wait to see what the director does next.",
    "I was hooked from the very first scene.",
    "The chemistry between the actors is palpable.",
    "This film is a triumph.",
    "I laughed, I cried, it was perfect.",
    "This movie has a powerful message.",
    "The visuals are stunning.",
    "I'm recommending this to everyone I know.",
    "This movie left me feeling inspired.",
    "I'm adding this to my list of all-time favorites.",
    "The storytelling is masterful.",
    "I was on the edge of my seat the entire time.",
    "This film stayed with me long after it ended.",
    "I'm in awe of the talent behind this movie.",
    "The pacing is perfect.",
    "I was thoroughly entertained from beginning to end.",
    "This movie is a cinematic gem.",
    "I'm already planning to watch it again.",
    "I haven't stopped talking about this movie since I saw it.",
    "The script is brilliant.",
    "I found myself rooting for the characters.",
    "This movie made me laugh out loud.",
    "The attention to detail is impressive.",
    "I was completely engrossed in the story.",
    "This film deserves all the awards.",
    "I was left speechless by the ending.",
    "I can't recommend this movie enough.",
    "The performances were Oscar-worthy.",
    "This movie is a masterpiece of storytelling.",
    "I was moved to tears by this film.",
    "The soundtrack is hauntingly beautiful.",
    "I was blown away by the cinematography.",
    "This film is a shining example of great cinema.",
    "I felt a deep connection to the characters.",
    "This movie restored my faith in cinema.",
    "I'm still processing how much I loved this movie.",
    "The direction is impeccable.",
    "I was genuinely surprised by the plot twists.",
    "This film is a rollercoaster of emotions.",
    "I'm recommending this to everyone I know.",
    "This movie is a must-watch for any film buff.",
    "I was captivated by every frame of this film.",
    "The attention to detail is extraordinary.",
    "I was completely absorbed in the story.",
    "This film is a true gem.",
    "I can't wait to watch it again.",
    "I was blown away by the performances.",
    "This movie has everything: action, suspense, heart.",
    "I was hooked from the very first scene.",
    "The chemistry between the actors is electric.",
    "This film is a triumph of storytelling.",
    "I laughed, I cried, it was an emotional rollercoaster.",
    "This movie stayed with me long after it ended.",
    "I'm recommending this to everyone I know.",
    "This movie deserves all the praise it's getting.",
    "I was on the edge of my seat the entire time.",
    "The script is brilliant.",
    "I found myself thinking about this movie for days afterward.",
    "The performances were outstanding.",
    "This film is a masterpiece of cinema.",
    "I was moved to tears by this movie.",
    "The soundtrack is phenomenal.",
    "This movie is a visual feast.",
    "I was captivated by the cinematography.",
    "This film is a must-see for any movie lover.",
    "I was blown away by the attention to detail.",
    "This movie is a true work of art.",
    "I was completely swept away by the story.",
    "This film is a testament to the power of cinema.",
    "I can't stop thinking about this movie.",
    "I was completely immersed in the world of the film.",
    "This movie is a game-changer for the industry.",
    "I was captivated by the performances.",
    "This film is a masterpiece of storytelling.",
    "I was deeply moved by the themes of this movie.",
    "The cinematography is breathtaking.",
    "This movie is a must-watch for anyone who loves great cinema.",
    "I was on the edge of my seat the entire time.",
    "The direction is flawless.",
    "This film is a tour de force.",
    "I was blown away by the performances.",
    "This movie is a triumph of cinema.",
    "I was deeply affected by this film.",
    "The soundtrack is incredible.",
    "This movie is a visual masterpiece.",
    "I was spellbound by the cinematography.",
    "This film is a true gem of modern cinema.",
    "I can't recommend this movie enough.",
    "I was completely engrossed in the story.",
    "This movie deserves all the accolades it's receiving.",
    "I was moved to tears by the ending.",
    "The performances were outstanding.",
    "This film is a masterpiece of cinema.",
    "I was deeply affected by this movie.",
    "The soundtrack is phenomenal.",
    "This movie is a visual feast.",
    "I was captivated by the cinematography.",
    "This film is a must-see for any movie lover.",
    "I was blown away by the attention to detail.",
    "This movie is a true work of art.",
    "I was completely swept away by the story.",
    "This film is a testament to the power of cinema.",
    "I can't stop thinking about this movie.",
    "I was completely immersed in the world of the film.",
    "This movie is a game-changer for the industry.",
    "I was captivated by the performances.",
    "This film is a masterpiece of storytelling.",
    "I was deeply moved by the themes of this movie.",
    "The cinematography is breathtaking.",
    "This movie is a must-watch for anyone who loves great cinema.",
    "I was on the edge of my seat the entire time.",
    "The direction is flawless.",
    "This film is a tour de force.",
    "This film was amazing.",
    "Super good movie.",
    "Such a good film. I'm happy.",
]
