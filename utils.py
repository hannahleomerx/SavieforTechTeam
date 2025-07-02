import os
import tantivy
from tantivy import Index


# Lade den Tantivy Index
def get_index(index_path) -> Index:
    if not os.path.exists(index_path):
        raise ValueError(f"Index path does not exist: {index_path}")

    try:
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("id", stored=True)
        schema_builder.add_text_field("url", stored=True)
        schema_builder.add_text_field("title", stored=True, tokenizer_name='en_stem')
        schema_builder.add_text_field("description", stored=True,
                                      tokenizer_name='en_stem')  # Multi-valued text field
        schema_builder.add_text_field("image", stored=True)
        schema_builder.add_integer_field("follower", stored=True, fast=True)
        schema_builder.add_integer_field("score", stored=True, fast=True)
        schema_builder.add_integer_field("start", stored=True, fast=True)
        schema_builder.add_text_field("locations", stored=True)
        schema_builder.add_text_field("countries", stored=True)
        schema_builder.add_text_field("genres", stored=True)
        schema_builder.add_integer_field("males", stored=True, fast=True)
        schema_builder.add_integer_field("females", stored=True, fast=True)
        schema_builder.add_integer_field("other", stored=True, fast=True)
        schema_builder.add_float_field("non_males", stored=True, fast=True)
        schema_builder.add_text_field("tmdb_overview", stored=True, tokenizer_name='en_stem')
        schema_builder.add_text_field("tmdb_poster_path", stored=True)
        schema_builder.add_integer_field("tmdb_genre_ids", stored=True, indexed=True)
        schema_builder.add_float_field("tmdb_popularity", stored=True, fast=True)
        schema_builder.add_float_field("tmdb_vote_average", stored=True, fast=True)
        schema_builder.add_integer_field("tmdb_vote_count", stored=True, fast=True)
        schema = schema_builder.build()
        index = tantivy.Index(schema, path=index_path)
        return index
    except Exception as e:
        raise RuntimeError(f"Failed to initialize index: {str(e)}")