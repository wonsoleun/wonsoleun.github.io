import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'zod';

const publications = defineCollection({
    loader: glob({ pattern: "**/*.md", base: "./src/content/publications" }),
    schema: z.object({
        title: z.string(),
        author: z.string().optional(),
        date: z.string().optional(),
        journal: z.string().optional(),
        external_url: z.string().optional(),
        image: z.string().optional(),
        description: z.string().optional(),
        tags: z.array(z.string()).optional(),
    }),
});

const talks = defineCollection({
    loader: glob({ pattern: "**/*.md", base: "./src/content/talks" }),
    schema: z.object({
        title: z.string(),
        date: z.string().optional(),
        event: z.string().optional(),
        external_url: z.string().optional(),
        description: z.string().optional(),
        tags: z.array(z.string()).optional(),
        image: z.string().optional(),
    }),
});

const posts = defineCollection({
    loader: glob({ pattern: "**/*.md", base: "./src/content/posts" }),
    schema: z.object({
        title: z.string(),
        date: z.string().optional(),
        description: z.string().optional(),
        author: z.string().optional(),
        tags: z.array(z.string()).optional(),
        external_url: z.string().optional(),
        image: z.string().optional(),
    }),
});

const teaching = defineCollection({
    loader: glob({ pattern: "**/*.md", base: "./src/content/teaching" }),
    schema: z.object({
        title: z.string(),
        institution: z.string().optional(),
        description: z.string().optional(),
        tags: z.array(z.string()).optional(),
        external_url: z.string().url().optional(),
    }),
});

const bio = defineCollection({
    loader: glob({ pattern: "bio.md", base: "./src/content" }),
    schema: z.object({
        name: z.string(),
        avatar: z.string(),
        shortBio: z.string().optional(),
        institution: z.string().optional(),
    }),
});

const projects = defineCollection({
    loader: glob({ pattern: "**/*.md", base: "./src/content/projects" }),
    schema: z.object({
        title: z.string(),
        description: z.string().optional(),
        tags: z.array(z.string()).optional(),
        external_url: z.string().optional(),
        image: z.string().optional(),
    }),
});

// 프로젝트별 하위 글 (예: src/content/project-posts/more-or-diverse/week1.md)
// 폴더 이름 = 프로젝트 md 파일 이름. 폴더에 md를 넣으면 해당 프로젝트 글목록에 자동 추가됨.
const projectPosts = defineCollection({
    loader: glob({ pattern: "**/*.md", base: "./src/content/project-posts" }),
    schema: z.object({
        title: z.string(),
        order: z.number().optional(),
        date: z.string().optional(),
        description: z.string().optional(),
        tags: z.array(z.string()).optional(),
    }),
});

const cv = defineCollection({
    loader: glob({ pattern: "cv.md", base: "./src/content" }),
    schema: z.object({
        name: z.string(),
        title: z.string(),
        experience: z.array(z.object({
            role: z.string(),
            institution: z.string(),
            period: z.string(),
            description: z.string(),
        })).optional(),
        education: z.array(z.object({
            degree: z.string(),
            institution: z.string(),
            period: z.string(),
            thesis: z.string().optional(),
            description: z.string().optional(),
        })).optional(),
    }),
});

export const collections = {
    'publications': publications,
    'talks': talks,
    'posts': posts,
    'bio': bio,
    'projects': projects,
    'projectPosts': projectPosts,
    'cv': cv,
    'teaching': teaching,
};
