# data and setting used for brashear_eval

prompts_dir = 'prompts/segments/'
expand = {'iw': 'irreducible-weak', 'is': 'irreducible-strong', 'rw': 'reducible-weak', 'rs': 'reducible-strong'}
# node names
names_brashear = ['James', 'Elizabeth', 'Henry', 'Anne',
         'Alyssa', 'Simon', 'Catherine', 'Thomas',
         'Peter', 'Michael', 'Lewis', 'Mary',
         'Victoria', 'Isabelle', 'Jane']
nodes_brashear = names_brashear

# ground-truth edges for irreducible weak and irrreducible strong
gt_i = [('James', 'Elizabeth'), ('Elizabeth', 'Henry'), ('James', 'Anne'),
           ('Alyssa', 'Simon'), ('Henry', 'Alyssa'), ('Catherine', 'Thomas'),
           ('Peter', 'Catherine'), ('Michael', 'Lewis'), ('Thomas', 'Lewis'),
           ('Mary', 'Catherine'), ('Victoria', 'Catherine'), ('Isabelle', 'Victoria'), ('Jane', 'Michael')]

# ground-truth edges for reducible weak and reducible strong
gt_r = [('James', 'Elizabeth'), ('James', 'Henry'), ('Elizabeth', 'Henry'),
           ('Henry', 'Anne'), ('Elizabeth', 'Alyssa'), ('James', 'Alyssa'),
           ('Simon', 'Alyssa'), ('Catherine', 'Thomas'), ('Catherine', 'Peter'),
           ('Catherine', 'Michael'), ('Catherine', 'Lewis'), ('Thomas', 'Peter'),
           ('Thomas', 'Michael'), ('Thomas', 'Lewis'), ('Peter', 'Lewis'),
           ('Michael', 'Lewis'), ('Peter', 'Michael'), ('Lewis', 'Mary'),
           ('Peter', 'Victoria'), ('Victoria', 'Jane'), ('Isabelle', 'Jane'),
        ('Victoria', 'Isabelle'), ('Alyssa','Henry')]


