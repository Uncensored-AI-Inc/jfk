create extension if not exists vector;

-- Create a table to store your JFK documents
create table
  jfk_documents (
    id uuid primary key,
    content text, -- corresponds to Document.pageContent
    metadata jsonb, -- corresponds to Document.metadata
    embedding vector (768) -- 768 works for OpenAI embeddings, change if needed
  );

-- Create a function to search for JFK documents
create function match_jfk_documents (
  query_embedding vector (768),
  filter jsonb default '{}'
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
) language plpgsql as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    content,
    metadata,
    1 - (jfk_documents.embedding <=> query_embedding) as similarity
  from jfk_documents
  where metadata @> filter
  order by jfk_documents.embedding <=> query_embedding;
end;
$$;
