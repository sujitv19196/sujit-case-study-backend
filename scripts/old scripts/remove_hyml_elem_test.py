from w3lib.html import remove_tags, remove_tags_with_content 

doc = '<div><p><head>This is a link:</head> <a href="http://www.example.com">example</a></p></div>'
print(remove_tags_with_content(doc, which_ones=('head',)))
