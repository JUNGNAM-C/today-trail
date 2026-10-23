a=open("../index_src.html",encoding="utf-8").read()
h='<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
open("index_gh.html","w",encoding="utf-8").write(h+a+"\n</html>\n")
