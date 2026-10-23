d=open('../deck.html',encoding='utf-8').read()
a=d.index('  var cards = [')
b=d.index('\n  ];\n',a)+len('\n  ];\n')
t=open('app_template.html',encoding='utf-8').read().replace('/*CARDS*/',d[a:b].rstrip('\n'))
old="    prog.last = i;\n    if (i > prog.max) prog.max = i;\n    saveProg();\n"
new="    if (app.getAttribute(\"data-view\") === \"read\") {\n      prog.last = i;\n      if (i > prog.max) prog.max = i;\n      saveProg();\n    }\n"
assert old in t
t=t.replace(old,new).replace("  paint(0); prog.last = prog.last || 0;\n  renderHome();","  paint(0);\n  renderHome();")
open('../index.html','w',encoding='utf-8').write(t)
