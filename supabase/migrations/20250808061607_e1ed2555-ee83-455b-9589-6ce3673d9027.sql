-- Common timestamp update function
create or replace function public.update_updated_at_column()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

-- PROFILES TABLE
create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  avatar_url text,
  bio text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.profiles enable row level security;

-- Profiles policies
create policy if not exists "Profiles are viewable by everyone"
  on public.profiles for select
  using (true);

create policy if not exists "Users can update their own profile"
  on public.profiles for update
  using (auth.uid() = id);

create policy if not exists "Users can insert their own profile"
  on public.profiles for insert
  with check (auth.uid() = id);

-- Trigger to keep updated_at fresh
create trigger if not exists update_profiles_updated_at
before update on public.profiles
for each row execute function public.update_updated_at_column();

-- When a new auth user signs up, insert a profile row
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = ''
as $$
begin
  insert into public.profiles (id, display_name, avatar_url, bio)
  values (new.id,
          coalesce(new.raw_user_meta_data ->> 'name', new.email),
          new.raw_user_meta_data ->> 'avatar_url',
          null)
  on conflict (id) do nothing;
  return new;
end;
$$;

create trigger if not exists on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();

-- POSTS TABLE
create table if not exists public.posts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  title text not null,
  content text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_posts_user_id on public.posts(user_id);
create index if not exists idx_posts_created_at on public.posts(created_at desc);

alter table public.posts enable row level security;

-- Posts policies: owner only
create policy if not exists "Users can view their own posts"
  on public.posts for select
  using (auth.uid() = user_id);

create policy if not exists "Users can insert their own posts"
  on public.posts for insert
  with check (auth.uid() = user_id);

create policy if not exists "Users can update their own posts"
  on public.posts for update
  using (auth.uid() = user_id);

create policy if not exists "Users can delete their own posts"
  on public.posts for delete
  using (auth.uid() = user_id);

create trigger if not exists update_posts_updated_at
before update on public.posts
for each row execute function public.update_updated_at_column();