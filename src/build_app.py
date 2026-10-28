d=open('trail-card-deck.html',encoding='utf-8').read()
a=d.index('  var cards = [')
b=d.index('\n  ];\n',a)+len('\n  ];\n')
c1=d[a:b].rstrip('\n').replace('var cards = [','var cards1 = [',1)
c2=open('deck2.js',encoding='utf-8').read().rstrip('\n')
t=open('app_template.html',encoding='utf-8').read().replace('/*CARDS*/',c1).replace('/*CARDS2*/',c2).replace('/*CARDS3*/',open('deck3.js',encoding='utf-8').read().rstrip('\n')).replace('/*CARDS4*/',open('deck4.js',encoding='utf-8').read().rstrip('\n')).replace('/*CARDS5*/',open('deck5.js',encoding='utf-8').read().rstrip('\n')).replace('/*CARDS6*/',open('deck6.js',encoding='utf-8').read().rstrip('\n'))
old="    prog.last = i;\n    if (i > prog.max) prog.max = i;\n    saveProg();\n"
new="    if (app.getAttribute(\"data-view\") === \"read\") {\n      prog.last = i;\n      if (i > prog.max) prog.max = i;\n      saveProg();\n    }\n"
assert old in t
t=t.replace(old,new).replace("  paint(0); prog.last = prog.last || 0;\n  renderHome();","  paint(0);\n  renderHome();")
open('app.html','w',encoding='utf-8').write(t)
