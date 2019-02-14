from bs4 import BeautifulSoup
import requests
import re

def retrieveText():
    print("Parsing text from online target")
    url = "https://www.whitehouse.gov/the-press-office/2017/10/16/remarks-president-trump-and-senate-majority-leader-mitch-mcconnell-joint"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "lxml")

    textwrapper = soup.find("div", { "class" : "field-item" })
    textel = textwrapper.find_all("p", { "class" : None })

    textstripped = []
    for element in textel:
        stripped = element.text.replace("\r", "\n").replace("\r", "").replace("\n", "").replace("Q ", "0002reporter: ").replace("THE PRESIDENT: ", "0001president: ").strip()
        if "P.M." not in stripped and "A.M." not in stripped:
            textstripped.append(stripped)


    # print(textstripped)
    return textstripped