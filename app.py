import pickle

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import ast

def main():
    st.title('IEEE Vis Paper Similarities')

    st.sidebar.markdown("## Overview ")
    st.sidebar.markdown("* Abstract document similarities for "
                        "3,500+ Vis papers (Vis, InfoVis, SciVis, DECISIVe, TVCG, and VAST).\n"
                        "* Similarities are the cosine similarity of the abstract's document embedding"
                        " (weighted by TF-IDF) using pre-trained GloVe word vectors (spaCy's `en_core_web_lg` model).\n"
                        "* Following Arora et al. (2017 ICML), we remove 1st PCA of doc embeddings to approx removing stop words.\n"
                        "* A UMAP representation provides the global topology of the document embedding space.")

    st.sidebar.markdown('**Dataset**')
    # dataset = st.sidebar.selectbox('',['Original','Scraped'])

    # outliers_flag = st.sidebar.checkbox("Remove outliers")
    df = load_data()

    # get paper titles
    # option_list=['DECISIVe','InfoVis','VAST','TVCG']

    papers = df.Title.unique()
    papers = papers[500:]

    # get keywords
    keywords = get_keywords()

    # load document embeddings
    docs_emb = load_embeddings()

    st.sidebar.markdown('**Approach**')
    approach = st.sidebar.selectbox('The approach chosen to query:', ['Embedding', 'Keyword'])

    if approach == "Embedding":

        st.sidebar.markdown('**Query Paper**')

        # Follow the below threads for updates on the Speed-Optimized search box option.
        # https://github.com/streamlit/streamlit/issues/1059
        # https://discuss.streamlit.io/t/streamlit-loading-column-data-takes-too-much-time/1791
        option = st.sidebar.multiselect('Selecting multiple will average the query vectors:',
                                        papers)  # index="The Anchoring Effect in Decision-Making with Visual Analytics"

        paper2 = df.index[df['Title'].isin(option)]

        color_choices = ['Similarity', 'Conference', 'Year']
        st.sidebar.markdown('**Plot Options**')
        color_option = st.sidebar.selectbox('Color', color_choices)
        brush_option = st.sidebar.checkbox('Brushing')

        # with st.spinner('Fetching the most similar papers...'):
        # st.success('Done!')
        results = get_similarities(df.index[df['Title'].isin(option)], docs_emb, df)
        select_num = st.sidebar.slider('Similar papers to return:', 1, 50, 20)
        # altair plot

        st.markdown('## Queried papers')
        # st.table(df[df['Title'].isin(option)].iloc[:, 0:3])
        df_filtered = df[df['Title'].isin(option)]
        st.table(df_filtered[["Title","Authors","Conference","Year"]])
        # show_umap = st.checkbox("UMAP View")

        if True:  # need to remove
            c = plot_scatter(df, query=option, highlight=results.head(n=select_num), color=color_option,
                             brush_option=brush_option)
            st.markdown("## UMAP Representation")
            # use alt.layer(c) to set height manually -- bug https://discuss.streamlit.io/t/alt-chart-height-is-being-removed/581/5
            st.altair_chart(alt.layer(c), width=0)

        # if show_keyword:
        # 	keyword = get_keywords(select_num)
        # 	st.markdown("## Keywords")
        # 	st.altair_chart(alt.layer(keyword), width=0)

        if brush_option:
            st.markdown("## Brushed documents")
            st.markdown("**UPDATE THIS WITH THE BRUSHED**")
        else:
            st.markdown("## Most similar documents")
            st.table(results.head(n=select_num))

        st.markdown("This app was created by [Ryan Wesslen](https://wesslen.netlify.app).")

    elif approach == "Keyword":

        st.sidebar.markdown('**Keywords**')
        select_keywords = st.sidebar.multiselect('Select Keywords:', keywords['Keyword'])
        select_num = st.sidebar.slider('Number of matching papers to return:', 1, 50, 20)

        # keyword_papers = df.index[df['Abstract'].str.match(select_keyword.value, case = False)]
        df["keyword_match"] = df.apply(lambda paper: filter_by_keywords(paper, select_keywords), axis=1)
        keyword_papers = df[df["keyword_match"] == True]

        c = plot_scatter(df, query=select_keywords, highlight=keyword_papers, color="Similarity",
                         brush_option=False)
        st.markdown("## UMAP Representation")
        # use alt.layer(c) to set height manually -- bug https://discuss.streamlit.io/t/alt-chart-height-is-being-removed/581/5
        st.altair_chart(alt.layer(c), width=0)

        st.markdown('## Queried papers')
        st.table(keyword_papers[["Title","Authors","Conference","Year"]].head(n=select_num))

    # if show_keyword:
    # 	keyword = get_keywords(select_num)
    # 	st.markdown("## Keywords")
    # 	st.altair_chart(alt.layer(keyword), width=0)


def load_data():
    df = pd.read_csv('data/vispapers-umap-updated.csv')

    return df

def filter_by_keywords(paper, keywords):
    try:
        paper_keywords_lower = [i.lower() for i in ast.literal_eval(paper["Keywords"])]
    except Exception as e:
        paper_keywords_lower = []
    filter_keywords_lower = [i.lower() for i in keywords]
    common_keywords = set(filter_keywords_lower).intersection(set(paper_keywords_lower))
    return len(common_keywords) > 0

# df = df[df.Conference.isin(['InfoVis','VAST','Vis','SciVis','DECISIVe','TVCG'])]
# if outliers_flag:
# 	df = df[df.Outliers==0]


def get_keywords():
    df = pd.read_csv('data/keyword_count.csv')
    return df.head(n=1000)


@st.cache
def load_embeddings():
    infile = open('data/docs_emb_updated_06272020', 'rb')
    docs_emb = pickle.load(infile)
    infile.close()

    return docs_emb


def cos_sim(v1, v2):
    cs = (np.dot(v1, v2)) / ((np.linalg.norm(v1)) * (np.linalg.norm(v2)))
    return cs


def get_sim(q, docs_emb, df):
    sims = []
    query = get_query(q, df, docs_emb)
    for i in range(len(docs_emb)):
        s = cos_sim(query, docs_emb[i])
        sims.append(s)

    zipped = zip(range(len(docs_emb)), sims)
    d = dict(zipped)

    sorted_x = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
    sim_df = pd.DataFrame(sorted_x, columns=['Index', 'Similarity'])  # modify to Index
    sim_df = sim_df.set_index('Index')
    return sim_df


def get_query(q, df, docs_emb):
    query_emb = np.mean(docs_emb[df.index.isin(q)], axis=0)
    return query_emb


@st.cache
def get_similarities(i, docs_emb, df):
    f = get_sim(i, docs_emb, df)
    ret = {"Similarity": f.Similarity,
           "Title": df.Title[f.index],
           "Authors": df.Authors[f.index],
           "Conference": df.Conference[f.index],
           "Year": df.Year[f.index],
           "Keywords": df.Keywords[f.index]
           }

    ret = pd.DataFrame(ret)
    ret = ret.drop(i)
    return ret


# @st.cache
# def get_keywords(df, )

# @st.cache
# def get_umap(df, n_neighbors = 5, min_dist = 0.1, metric = 'cosine'):
# 	viz = umap.UMAP(
# 	    random_state=42, 
# 	    n_neighbors=n_neighbors, 
# 	    min_dist=min_dist, 
# 	    n_components=2, 
# 	    metric=metric).fit_transform(df)
# 	return viz

def plot_scatter(df, query=None, highlight=None, color="Similarity", brush_option=False):
    query_index = df.index[df['Title'].isin(query)]

    df['Highlight'] = df.index.isin(highlight.index)
    df.Highlight[df.index.isin(query_index)] = -2

    # sort by df['Highlight'] so that red and blue are on top

    domain = [1, 0, -2]
    range_ = ['red', 'darkgrey', 'blue']
    size = 10
    alpha = 0.4

    # Brush for selection
    # brush = alt.selection(type='interval')
    if color == "Similarity":
        points = alt.Chart(df).mark_circle(size=size, opacity=alpha).encode(
            x='X',
            y='Y',
            color=alt.Color('Highlight:N', scale=alt.Scale(domain=domain, range=range_), legend=None),
            # remove legend
            # color = alt.Color(alt.condition(brush, 'Highlight:N', alt.value('grey')), scale=alt.Scale(domain=domain, range=range_)),
            tooltip=['Title', 'Conference', 'Authors', 'Year']
        )

    elif color == "Conference":
        points = alt.Chart(df).mark_circle(size=size, opacity=alpha).encode(
            x='X',
            y='Y',
            color=alt.Color('Conference:N'),  # remove legend
            # color = alt.Color(alt.condition(brush, 'Highlight:N', alt.value('grey')), scale=alt.Scale(domain=domain, range=range_)),
            tooltip=['Title', 'Conference', 'Authors', 'Year']
        )

    elif color == "Year":
        points = alt.Chart(df).mark_circle(size=size, opacity=alpha).encode(
            x='X',
            y='Y',
            color=alt.Color('Year:O'),  # remove legend
            # color = alt.Color(alt.condition(brush, 'Highlight:N', alt.value('grey')), scale=alt.Scale(domain=domain, range=range_)),
            tooltip=['Title', 'Conference', 'Authors', 'Year']
        )

    if brush_option:
        brush = alt.selection(type='interval')

        points = points.add_selection(brush).properties(height=500)

        ## need to pass this
        ranked_text = alt.Chart(df).mark_text().encode(
            y=alt.Y('row_number:O', axis=None)
        ).transform_window(
            row_number='row_number()'
        ).transform_filter(
            brush
        ).transform_window(
            rank='rank(row_number)'
        ).transform_filter(
            alt.datum.rank < 10
        )

    else:
        points = points.interactive().properties(height=500)

    # # Data Tables
    # title = ranked_text.encode(text='Title:N').properties(title='Title')
    # #authors = ranked_text.encode(text='Authors:N').properties(title='Authors')
    # conference = ranked_text.encode(text='Conference:N').properties(title='Conference')
    # year = ranked_text.encode(text='Year:N').properties(title='Year')

    # text = alt.hconcat(title, conference, year) # Combine data tables

    # # Build chart
    # charts = alt.vconcat(
    #     points,
    #     text
    # ).resolve_legend(
    #     color="independent"
    # )

    # .properties(
    #     width=800,
    #     height=500
    # )

    # Base chart for data tables
    # ranked_text = alt.Chart(df).mark_text().encode(
    #     y=alt.Y('row_number:O',axis=None)
    # ).transform_window(
    #     row_number='row_number()'
    # ).transform_filter(
    #     brush
    # ).transform_window(
    #     rank='rank(row_number)'
    # ).transform_filter(
    #     alt.datum.rank<10
    # )

    # Data Tables
    # title = ranked_text.encode(text='Title:N').properties(title='Title')
    # #authors = ranked_text.encode(text='Authors:N').properties(title='Authors')
    # conference = ranked_text.encode(text='Conference:N').properties(title='Conference')
    # year = ranked_text.encode(text='Year:N').properties(title='Year')

    # text = alt.hconcat(title, conference, year) # Combine data tables

    # # Build chart
    # chart = alt.vconcat(
    #     points,
    #     text
    # ).resolve_legend(
    #     color="independent"
    # )

    return points


# from gensim.corpora import Dictionary
# from gensim.models.tfidfmodel import TfidfModel
# from gensim.matutils import sparse2full
# import spacy
# nlp = spacy.load("en_core_web_lg", disable=("parser", "tagger", "ner"))


if __name__ == "__main__":
    main()
