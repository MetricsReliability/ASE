from github import Github

g = Github("nimashiri", "Nim@13700")
user = g.get_user()


if __name__ == '__main__':
    print("this program is being run by itself")
else:
    print("imported from another program")