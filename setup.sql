
CREATE TABLE redditcomment (
    label int,
    researched_by varchar,
    text varchar NOT NULL,
    title varchar,
    url varchar,
    comments varchar,
    comment_author varchar,
    comment_text varchar,
    comment_score float,
    comment_subreddit varchar,
    CONSTRAINT redditcomment_pkey PRIMARY KEY(title,comment_text)
);

CREATE TABLE factcheck (
    text varchar PRIMARY KEY,
    date varchar,
    author_type varchar,
    author varchar,
    url varchar,
    rating_type varchar,
    rating varchar,
    datafeedelement varchar,
    language varchar
);

CREATE TABLE topic (
	title varchar PRIMARY KEY,
    topic varchar,
    perception float
);

