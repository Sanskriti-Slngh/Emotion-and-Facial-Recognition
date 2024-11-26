import random

numberoftry = 10
name = input("What is your name?\n")
a = print("Hello, " + name + " let's play hangman. Let me tell you a clue its christmas themed and you have 10 tries. :)")
words = ["merry", "christmas", "dad", "favorite", "santa", "fantastic", "family", "best", "caring", "sharing", "kirti", "manish", "briti", "sanskriti", "friends", "holly", "jolly", "snowy", "candle", "fire", "cookies", "chimney", "gifts", "presents", "tree", "fun", "candlelit", "festive", "mistletoe", "lights", "decoration", "bulb", "celebration", "seasonal", "spiritual", "ho", "snowman", "snow", "ice", "icicles", "Advent", "angels","announcement","bells","candy","cards","cedar","cold","comet","crowds","dancer","dasher","december","dolls","donner","dressing","eggnog","elves","fir","frosty","fruitcake","goodwill","greetings","lists","merry","miracle","mistletoe","noel","pageant","parades","party","pie","pine","poinsettia","prancer","presents","pie","punch","green","reindeer","ribbon","rudolph","sacred","sales","sauce","scrooge","season","sled","snowflakes","spirit","stand","star","stickers","tidings","tinsel","togetherness","toys","tradition","traffic","trips","turkey","vacation","vixen","winter", "worship", "wrapping paper", "wreath", "yule", "yuletide"]
word = random.choice(words)
wordlist = list(word)
print(len(wordlist))

for i in range(numberoftry):
    guess = input("Please guess a letter, only one character?    By the way all letters are LOWERCASED and you must type every letter even if repeated.\n")
    q = [guess]
    if guess in wordlist:
        print ("You got a letter! " + str(numberoftry) + " tries left.")
        now = wordlist.remove(guess)
        if len(wordlist) == 0:
            print("You won!!!")
            print("The word was " + word + ".")
            exit()
    else:
        numberoftry = numberoftry - 1
        print("sorry, that is wrong, only " + str(numberoftry) + " tries left.")
        if len(wordlist) == 0:
            print("You won!!!")
            print("The word was " + word + ".")
            exit()
print("The word was " + word + ".")