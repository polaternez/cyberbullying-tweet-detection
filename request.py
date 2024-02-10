import requests


url = "http://localhost:8080/predict_api"
data = {
    "input": [
    'Where. Is. The. Logic? "@kenny_Gurl: @BrendanRochford @YesYoureRacist all niggers smell like shit and have dumb nigger names fuck niggers"',
    'Females is always used as a replacement for bitch, whenever you see niggas or even women using females its in a negative connotation so lately I’ll tell niggas if they wanna call me a bitch they should just say that',
    '@slagkick frost for now. still learning how to play. for bgs, seems to work well.',
    'Which will end first: #mkr or Tony Abbott as PM?'
    ]
}
r = requests.post(url, json=data)

print(r.text)
