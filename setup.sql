CREATE TABLE redditnews (
    label int,
    researched_by varchar,
    text varchar NOT NULL,
    title varchar,
    url varchar,
    comments varchar
);

CREATE TABLE redditcomment (
    text varchar NOT NULL,
    title varchar,
    comment_author varchar,
    comment_text varchar NOT NULL,
    comment_score int,
    comment_subreddit varchar
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
	title varchar,
    topic varchar,
    perception float
);

